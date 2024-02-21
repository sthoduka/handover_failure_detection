Computes the optical flow for a video using the TV-L1 algorithm.

### Build
    mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make
### Run
    cd build
    ./calculate_optical_flow <path to video file> <output directory for flow images>
