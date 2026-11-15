// Runs dlib's shape_predictor on raw images and boxes, printing the parts, so the effect of
// build options (FMA contraction, LAPACK) on dlib's own landmarks can be measured.
//
//   sp_probe MODEL.dat CASES.txt > parts.txt
//
// CASES.txt: one case per line: `RAW_PATH HEIGHT WIDTH CHANNELS LEFT TOP RIGHT BOTTOM`, where
// RAW_PATH holds HEIGHT*WIDTH*CHANNELS bytes (C order). Output: one line per case with
// 2*num_parts integers.
//
// Build (no CMake needed), e.g. without FMA contraction and without LAPACK:
//   c++ -std=c++17 -O2 -ffp-contract=off -DDLIB_NO_GUI_SUPPORT -I third_party/dlib \
//       tools/cpp_probe/sp_probe.cpp third_party/dlib/dlib/all/source.cpp -o sp_probe_nofma
// or like dlib-bin on macOS (Accelerate):
//   ... -ffp-contract=on -DDLIB_USE_BLAS -DDLIB_USE_LAPACK ... -framework Accelerate
#include <dlib/image_processing.h>

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

using namespace dlib;

int main(int argc, char** argv)
{
    if (argc != 3) { std::cerr << "usage: sp_probe MODEL CASES\n"; return 2; }
    shape_predictor sp;
    deserialize(argv[1]) >> sp;
    std::ifstream cases(argv[2]);
    std::string line;
    std::map<std::string, std::vector<unsigned char>> cache;
    while (std::getline(cases, line))
    {
        std::istringstream in(line);
        std::string path; long h, w, c, l, t, r, b;
        in >> path >> h >> w >> c >> l >> t >> r >> b;
        auto& buf = cache[path];
        if (buf.empty())
        {
            std::ifstream f(path, std::ios::binary);
            buf.assign(std::istreambuf_iterator<char>(f), {});
        }
        full_object_detection det;
        const rectangle rect(l, t, r, b);
        if (c == 1)
        {
            array2d<unsigned char> img(h, w);
            for (long y = 0; y < h; ++y) for (long x = 0; x < w; ++x) img[y][x] = buf[y*w + x];
            det = sp(img, rect);
        }
        else
        {
            array2d<rgb_pixel> img(h, w);
            for (long y = 0; y < h; ++y) for (long x = 0; x < w; ++x)
                img[y][x] = rgb_pixel(buf[(y*w + x)*3], buf[(y*w + x)*3 + 1], buf[(y*w + x)*3 + 2]);
            det = sp(img, rect);
        }
        for (unsigned long i = 0; i < det.num_parts(); ++i)
            std::cout << det.part(i).x() << ' ' << det.part(i).y() << (i + 1 < det.num_parts() ? ' ' : '\n');
    }
    return 0;
}
