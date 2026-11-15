// Writes the serialized frontal face detector embedded in dlib's
// frontal_face_detector.h (base64 + compress_stream::kernel_1ea) to a file,
// byte for byte, so manyfaced can embed it without porting dlib's compressor.
//
// Build (no CMake needed, only dlib headers and two dlib sources):
//   c++ -std=c++14 -O1 -I third_party/dlib \
//       tools/cpp_probe/extract_frontal_detector.cpp \
//       third_party/dlib/dlib/base64/base64_kernel_1.cpp \
//       -o tools/cpp_probe/build/extract_frontal_detector
#include <dlib/image_processing/frontal_face_detector.h>

#include <fstream>
#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "usage: " << argv[0] << " OUTPUT.dat\n";
        return 2;
    }
    const std::string data = dlib::get_serialized_frontal_faces();
    // Round-trip through dlib's own deserializer to make sure the bytes are valid.
    std::istringstream sin(data);
    dlib::frontal_face_detector detector;
    dlib::deserialize(detector, sin);
    std::ofstream out(argv[1], std::ios::binary);
    out.write(data.data(), data.size());
    std::cerr << "wrote " << data.size() << " bytes, " << detector.num_detectors() << " filters\n";
    return out ? 0 : 1;
}
