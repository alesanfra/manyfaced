// Dumps the intermediates of dlib's frontal face detector for one raw image, so the Rust port
// can be compared stage by stage:
//
//   hog_probe IMAGE.raw OUT_DIR [upsample] [adjust_threshold]
//
// IMAGE.raw: int32 width, int32 height, int32 channels (1 or 3), then row-major uint8 pixels.
// OUT_DIR receives .npy files:
//   up_image.npy                     the image after pyramid_up (if upsample > 0)
//   level{L}_image.npy               pyramid_down<6> level L (level 0 is the input)
//   level{L}_fhog.npy                31 x H x W float FHOG planes of level L (with filter padding)
//   filter{I}_full.npy               31 x 10 x 10 float filters of detector I
//   filter{I}_rank.npy               int32 per plane: number of separable components
//   filter{I}_row.npy / _col.npy     float, concatenated separable components (plane order)
//   filter{I}_separable.npy          int32 [1] if the separable path is used
//   level{L}_sal{I}.npy              float saliency image of detector I on level L, plus
//   level{L}_area{I}.npy             int64 [l, t, r, b] of the valid area
//   dets.npy                         float64 rows [l, t, r, b, score, weight_index] (final)
//
// Build (no CMake needed; add -DDLIB_USE_BLAS -DDLIB_USE_LAPACK -framework Accelerate and/or
// -ffp-contract=off|on for the build variants discussed in docs/internals.md):
//   c++ -std=c++17 -O2 -DDLIB_NO_GUI_SUPPORT -I third_party/dlib \
//       tools/cpp_probe/hog_probe.cpp third_party/dlib/dlib/all/source.cpp \
//       -o tools/cpp_probe/build/hog_probe
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_transforms.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace dlib;

static void write_npy(const std::string& path, const char* descr, const std::vector<long>& shape,
                      const void* data, size_t nbytes)
{
    std::ostringstream h;
    h << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) h << shape[i] << (shape.size() == 1 || i + 1 < shape.size() ? ", " : "");
    h << "), }";
    std::string header = h.str();
    size_t total = 10 + header.size() + 1;
    header.append((64 - total % 64) % 64, ' ');
    header.push_back('\n');
    std::ofstream out(path, std::ios::binary);
    out.write("\x93NUMPY\x01\x00", 8);
    uint16_t len = header.size();
    out.write(reinterpret_cast<const char*>(&len), 2);
    out << header;
    out.write(reinterpret_cast<const char*>(data), nbytes);
}

template <typename pixel>
static void dump_image(const std::string& path, const array2d<pixel>& img)
{
    std::vector<unsigned char> buf;
    for (long r = 0; r < img.nr(); ++r)
        for (long c = 0; c < img.nc(); ++c)
        {
            const unsigned char* p = reinterpret_cast<const unsigned char*>(&img[r][c]);
            buf.insert(buf.end(), p, p + sizeof(pixel));
        }
    std::vector<long> shape = {img.nr(), img.nc()};
    if (sizeof(pixel) == 3) shape.push_back(3);
    write_npy(path, "|u1", shape, buf.data(), buf.size());
}

static void dump_float(const std::string& path, const array2d<float>& a)
{
    std::vector<float> buf;
    for (long r = 0; r < a.nr(); ++r)
        for (long c = 0; c < a.nc(); ++c) buf.push_back(a[r][c]);
    write_npy(path, "<f4", {a.nr(), a.nc()}, buf.data(), buf.size() * 4);
}

template <typename pixel>
static int run(array2d<pixel>& input, const std::string& out, unsigned upsample, double adjust)
{
    frontal_face_detector det = get_frontal_face_detector();
    const auto& scanner = det.get_scanner();
    pyramid_down<2> pyr2;
    array2d<pixel> img;
    if (upsample > 0)
    {
        pyramid_up(input, img, pyr2);
        for (unsigned i = 1; i < upsample; ++i) pyramid_up(img);
        dump_image(out + "/up_image.npy", img);
    }
    else
    {
        assign_image(img, input);
    }

    // Pyramid levels exactly like impl::create_fhog_pyramid.
    unsigned long width = scanner.get_fhog_window_width(), height = scanner.get_fhog_window_height();
    array<array<array2d<float>>> feats;
    impl::create_fhog_pyramid<pyramid_down<6>>(img, default_fhog_feature_extractor(), feats, 8, height, width,
                                                 scanner.get_min_pyramid_layer_width(), scanner.get_min_pyramid_layer_height(),
                                                 scanner.get_max_pyramid_levels());
    {
        pyramid_down<6> pyr;
        array2d<pixel> a, b;
        assign_image(a, img);
        for (unsigned long l = 0; l < feats.size(); ++l)
        {
            dump_image(out + "/level" + std::to_string(l) + "_image.npy", a);
            pyr(a, b);
            swap(a, b);
        }
    }
    for (unsigned long l = 0; l < feats.size(); ++l)
    {
        std::vector<float> buf;
        long nr = feats[l][0].nr(), nc = feats[l][0].nc();
        for (unsigned long p = 0; p < feats[l].size(); ++p)
            for (long r = 0; r < nr; ++r)
                for (long c = 0; c < nc; ++c) buf.push_back(feats[l][p][r][c]);
        write_npy(out + "/level" + std::to_string(l) + "_fhog.npy", "<f4", {(long)feats[l].size(), nr, nc}, buf.data(), buf.size() * 4);
    }

    for (unsigned long i = 0; i < det.num_detectors(); ++i)
    {
        auto fb = scanner.build_fhog_filterbank(det.get_w(i));
        std::vector<float> full, rows, cols;
        std::vector<int32_t> rank;
        for (unsigned long p = 0; p < fb.filters.size(); ++p)
        {
            for (long r = 0; r < fb.filters[p].nr(); ++r)
                for (long c = 0; c < fb.filters[p].nc(); ++c) full.push_back(fb.filters[p](r, c));
            rank.push_back(fb.row_filters[p].size());
            for (auto& v : fb.row_filters[p]) for (long k = 0; k < v.size(); ++k) rows.push_back(v(k));
            for (auto& v : fb.col_filters[p]) for (long k = 0; k < v.size(); ++k) cols.push_back(v(k));
        }
        const std::string pre = out + "/filter" + std::to_string(i);
        write_npy(pre + "_full.npy", "<f4", {(long)fb.filters.size(), fb.filters[0].nr(), fb.filters[0].nc()}, full.data(), full.size() * 4);
        write_npy(pre + "_rank.npy", "<i4", {(long)rank.size()}, rank.data(), rank.size() * 4);
        write_npy(pre + "_row.npy", "<f4", {(long)rows.size()}, rows.data(), rows.size() * 4);
        write_npy(pre + "_col.npy", "<f4", {(long)cols.size()}, cols.data(), cols.size() * 4);
        int32_t sep = fb.num_separable_filters() > fb.filters.size() * std::min(fb.filters[0].nr(), fb.filters[0].nc()) / 3.0 ? 0 : 1;
        write_npy(pre + "_separable.npy", "<i4", {1}, &sep, 4);
        for (unsigned long l = 0; l < feats.size(); ++l)
        {
            array2d<float> sal;
            rectangle area = impl::apply_filters_to_fhog(fb, feats[l], sal);
            dump_float(out + "/level" + std::to_string(l) + "_sal" + std::to_string(i) + ".npy", sal);
            int64_t a[4] = {area.left(), area.top(), area.right(), area.bottom()};
            write_npy(out + "/level" + std::to_string(l) + "_area" + std::to_string(i) + ".npy", "<i8", {4}, a, 32);
        }
    }

    std::vector<rect_detection> dets;
    det(img, dets, adjust);
    std::vector<double> rows;
    for (auto& d : dets)
    {
        rectangle r = d.rect;
        if (upsample > 0) r = pyr2.rect_down(r, upsample);
        rows.insert(rows.end(), {(double)r.left(), (double)r.top(), (double)r.right(), (double)r.bottom(),
                                 d.detection_confidence, (double)d.weight_index});
    }
    write_npy(out + "/dets.npy", "<f8", {(long)dets.size(), 6}, rows.data(), rows.size() * 8);
    std::cerr << feats.size() << " levels, " << dets.size() << " detections\n";
    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "usage: hog_probe IMAGE.raw OUT_DIR [upsample] [adjust_threshold]\n";
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    int32_t hdr[3];
    in.read(reinterpret_cast<char*>(hdr), 12);
    const unsigned upsample = argc > 3 ? std::stoul(argv[3]) : 0;
    const double adjust = argc > 4 ? std::stod(argv[4]) : 0.0;
    if (hdr[2] == 1)
    {
        array2d<unsigned char> img(hdr[1], hdr[0]);
        in.read(reinterpret_cast<char*>(&img[0][0]), (long)hdr[0] * hdr[1]);
        return run(img, argv[2], upsample, adjust);
    }
    array2d<rgb_pixel> img(hdr[1], hdr[0]);
    in.read(reinterpret_cast<char*>(&img[0][0]), (long)hdr[0] * hdr[1] * 3);
    return run(img, argv[2], upsample, adjust);
}
