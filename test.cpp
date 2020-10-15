#include "lodepng.h"
#include <iostream>

#include "timer.h"


int main(int argc, char *argv[]) {
    Timer t;
    std::vector<unsigned char> image; //the raw pixels
    unsigned width, height;

    //decode
    unsigned error = lodepng::decode(image, width, height, "save.png");
    std::cout << image.size() << " " << width << " " << height << std::endl;
    t.end();
    t.start();
    lodepng::encode("test.png", image, width, height);
    t.end();
    return 0;

}