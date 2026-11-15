// Checks dlib's non-separable FHOG filtering path (used by detectors whose filters are not low
// rank): builds a filter bank from arbitrary weights and dumps the level-0 saliency image.
//
//   hog_full_probe IMAGE.raw WEIGHTS.f64 OUT.npy
//
// WEIGHTS.f64: 3101 little-endian doubles (31 planes of 10x10 filters, then the threshold),
// for an 80x80 window, cell size 8, padding 0 (the frontal face detector's geometry).
//
// Build like hog_probe.cpp:
//   c++ -std=c++17 -O2 -DDLIB_NO_GUI_SUPPORT -I third_party/dlib \
//       tools/cpp_probe/hog_full_probe.cpp third_party/dlib/dlib/all/source.cpp \
//       -o tools/cpp_probe/build/hog_full_probe
#include <dlib/image_processing/frontal_face_detector.h>

#include <cstdint>
#include <fstream>
#include <iostream>

using namespace dlib;

int main(int argc, char** argv)
{
    if (argc != 4) { std::cerr << "usage: hog_full_probe IMAGE.raw WEIGHTS.f64 OUT.npy\n"; return 2; }
    std::ifstream in(argv[1], std::ios::binary);
    int32_t hdr[3];
    in.read(reinterpret_cast<char*>(hdr), 12);
    if (hdr[2] != 1) { std::cerr << "grayscale only\n"; return 2; }
    array2d<unsigned char> img(hdr[1], hdr[0]);
    in.read(reinterpret_cast<char*>(&img[0][0]), (long)hdr[0] * hdr[1]);

    matrix<double, 0, 1> w(3101);
    std::ifstream win(argv[2], std::ios::binary);
    win.read(reinterpret_cast<char*>(&w(0)), 3101 * 8);

    frontal_face_detector det = get_frontal_face_detector();
    const auto& scanner = det.get_scanner();
    auto fb = scanner.build_fhog_filterbank(w);
    const bool full = fb.num_separable_filters() > fb.filters.size() * std::min(fb.filters[0].nr(), fb.filters[0].nc()) / 3.0;
    std::cerr << (full ? "full" : "separable") << " path, " << fb.num_separable_filters() << " separable filters\n";

    array<array2d<float>> feats;
    extract_fhog_features(img, feats, 8, 10, 10);
    array2d<float> sal;
    rectangle area = impl::apply_filters_to_fhog(fb, feats, sal);
    std::cerr << "area " << area << "\n";

    std::ofstream out(argv[3], std::ios::binary);
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(sal.nr()) + ", " + std::to_string(sal.nc()) + "), }";
    header.append((64 - (10 + header.size() + 1) % 64) % 64, ' ');
    header.push_back('\n');
    out.write("\x93NUMPY\x01\x00", 8);
    uint16_t len = header.size();
    out.write(reinterpret_cast<const char*>(&len), 2);
    out << header;
    for (long r = 0; r < sal.nr(); ++r) out.write(reinterpret_cast<const char*>(&sal[r][0]), sal.nc() * 4);
    return 0;
}
