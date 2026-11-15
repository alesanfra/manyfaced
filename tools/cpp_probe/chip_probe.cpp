// Computes dlib::get_face_chip_details for landmark sets read from a text file, printing the
// results with full precision, to find where manyfaced's chip details diverge from dlib's.
//
// Input lines: num_parts size padding x0 y0 x1 y1 ...
// Output lines: left top right bottom angle rows cols (%.17g)
//
// Build (no CMake needed; vary -ffp-contract to test FMA contraction):
//   c++ -std=c++17 -O2 -ffp-contract=off -DDLIB_NO_GUI_SUPPORT -I third_party/dlib \
//       tools/cpp_probe/chip_probe.cpp -o tools/cpp_probe/build/chip_probe
#include <dlib/image_transforms/interpolation.h>
#include <dlib/image_processing/full_object_detection.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

int main(int argc, char** argv)
{
    if (argc != 2) { std::cerr << "usage: chip_probe CASES.txt\n"; return 2; }
    std::ifstream in(argv[1]);
    std::string line;
    while (std::getline(in, line))
    {
        std::istringstream s(line);
        unsigned long n, size;
        double padding;
        s >> n >> size >> padding;
        std::vector<dlib::point> parts(n);
        for (auto& p : parts) s >> p.x() >> p.y();
        dlib::full_object_detection det(dlib::rectangle(0, 0, 10, 10), parts);
        const dlib::chip_details d = dlib::get_face_chip_details(det, size, padding);
        std::printf("%.17g %.17g %.17g %.17g %.17g %lu %lu\n", d.rect.left(), d.rect.top(),
                    d.rect.right(), d.rect.bottom(), d.angle, d.rows, d.cols);
    }
    return 0;
}
