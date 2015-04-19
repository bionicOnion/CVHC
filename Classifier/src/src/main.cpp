#include "NeuralNet.h"
#include "CharacterExtractor.h"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>

int main(int argc, char** argv)
{
  if (argc < 3)
  {
    printf("Usage: %s [classifier] [image]\n", argv[0]);
    return 0;
  }
  
  NeuralNet nn;
  CharacterExtractor ce;

  if (!nn.loadNN(argv[1]))
  {
    fprintf(stderr, "There were errors while loading the neural network.\n");
    return 0;
  }

  cv::Mat srcImage;
  srcImage = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
  if (!srcImage.data)
  {
    printf("The specified image could not be found.\n");
    return 0;
  }

  int errCode;
  cv::Mat destImage, threshImage, croppedImage;
  std::vector<cv::Rect> boundingBoxes, charBoundingBoxes;
  if ((errCode = ce.preprocessImage(srcImage, destImage, threshImage, false)) != 0)
  {
    printf("Preprocessing failed with code %i\n", errCode);
    return errCode;
  }
  if ((errCode = ce.findBoundingBoxes(destImage, boundingBoxes)) != 0)
  {
    printf("Finding bounding boxes failed with code %i\n", errCode);
    return errCode;
  }
  if ((errCode = ce.findFullCharBoxes(boundingBoxes, charBoundingBoxes)) != 0)
  {
    printf("Correcting bounding boxes failed with code %i\n", errCode);
    return errCode;
  }
  std::vector<cv::Rect>::iterator iter;
  for (iter = charBoundingBoxes.begin(); iter != charBoundingBoxes.end(); ++iter)
  {
    if ((errCode = ce.cropImage(threshImage, croppedImage, *iter)) != 0)
    {
      printf("Cropping failed with code %i\n", errCode);
      return errCode;
    }

    char label = nn.classify(croppedImage);
    printf("Prediction: %c\n", label);
  }
  return 0;
}