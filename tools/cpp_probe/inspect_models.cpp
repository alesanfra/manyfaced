// Prints structural facts and checksums of dlib models, as computed by dlib itself, so the
// Rust parser can be cross-checked: inspect_models SP5.dat SP68.dat RESNET.dat
//
// Build (no CMake needed):
//   c++ -std=c++17 -O2 -DDLIB_NO_GUI_SUPPORT -I third_party/dlib \
//       tools/cpp_probe/inspect_models.cpp third_party/dlib/dlib/all/source.cpp \
//       -o tools/cpp_probe/build/inspect_models
#include <dlib/dnn.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include <cstdio>
#include <iostream>

using namespace dlib;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;
template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;
template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;
template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;
using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<alevel1<alevel2<alevel3<alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>>>>>>>>>>>>>;

static double sum_tensor(const tensor& t)
{
    double s = 0;
    for (auto v : t) s += v;
    return s;
}

int main(int argc, char** argv)
{
    if (argc != 4) { std::cerr << "usage: inspect_models SP5 SP68 RESNET\n"; return 2; }
    for (int i = 1; i <= 2; ++i)
    {
        shape_predictor sp;
        deserialize(argv[i]) >> sp;
        std::printf("shape_predictor %s parts=%lu features=%lu\n", argv[i], sp.num_parts(), sp.num_features());
    }
    anet_type net;
    deserialize(argv[3]) >> net;
    std::printf("resnet params=%zu\n", count_parameters(net));
    // Checksums of a few layers, in execution order: layer<130> is the stem con,
    // layer<129> the stem affine, layer<1> the fc.
    auto& stem_con = layer<130>(net).layer_details();
    std::printf("stem_con params_sum=%.9g first=%.9g\n", sum_tensor(stem_con.get_layer_params()), *stem_con.get_layer_params().host());
    auto& stem_affine = layer<129>(net).layer_details();
    std::printf("stem_affine gamma0=%.9g beta0=%.9g\n", stem_affine.get_gamma().host()[0], stem_affine.get_beta().host()[0]);
    auto& fc = layer<1>(net).layer_details();
    std::printf("fc params_sum=%.9g\n", sum_tensor(fc.get_layer_params()));

    frontal_face_detector det = get_frontal_face_detector();
    std::printf("frontal detectors=%lu dims=%ld\n", det.num_detectors(), det.get_scanner().get_num_dimensions());
    return 0;
}
