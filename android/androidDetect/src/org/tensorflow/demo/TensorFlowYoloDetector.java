/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.demo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;
import android.util.Log;


import org.tensorflow.demo.SqueezeNcnn;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

/** An object detector that uses TF and a YOLO model to detect objects. */
public class TensorFlowYoloDetector implements Classifier {


  // Only return this many results with at least this confidence.
  private static final int MAX_RESULTS = 5;

  private static final int NUM_CLASSES = 80;

  private static final int NUM_BOXES_PER_BLOCK = 5;

  // TODO(andrewharp): allow loading anchors and classes
  // from files.
  private static final double[] ANCHORS = {
          0.738768,0.874946,  2.42204,2.65704,  4.30971,7.04493,  10.246,4.59428,  12.6868,11.8741
  };

  private static final String[] LABELS = {
          "person",
          "bicycle",
          "car",
          "motorbike",
          "aeroplane",
          "bus",
          "train",
          "truck",
          "boat",
          "traffic light",
          "fire hydrant",
          "stop sign",
          "parking meter",
          "bench",
          "bird",
          "cat",
          "dog",
          "horse",
          "sheep",
          "cow",
          "elephant",
          "bear",
          "zebra",
          "giraffe",
          "backpack",
          "umbrella",
          "handbag",
          "tie",
          "suitcase",
          "frisbee",
          "skis",
          "snowboard",
          "sports ball",
          "kite",
          "baseball bat",
          "baseball glove",
          "skateboard",
          "surfboard",
          "tennis racket",
          "bottle",
          "wine glass",
          "cup",
          "fork",
          "knife",
          "spoon",
          "bowl",
          "banana",
          "apple",
          "sandwich",
          "orange",
          "broccoli",
          "carrot",
          "hot dog",
          "pizza",
          "donut",
          "cake",
          "chair",
          "sofa",
          "pottedplant",
          "bed",
          "diningtable",
          "toilet",
          "tvmonitor",
          "laptop",
          "mouse",
          "remote",
          "keyboard",
          "cell phone",
          "microwave",
          "oven",
          "toaster",
          "sink",
          "refrigerator",
          "book",
          "clock",
          "vase",
          "scissors",
          "teddy bear",
          "hair drier",
          "toothbrush"
  };
  //
  private static  void initSqueezeNcnn(final AssetManager assetManager,SqueezeNcnn squeezencnn) throws IOException
  {
    byte[] param = null;
    byte[] bin = null;
    byte[] words = null;

    {
      InputStream assetsInputStream = assetManager.open("ncnnNet.param.bin");
      int available = assetsInputStream.available();
      param = new byte[available];
      int byteCode = assetsInputStream.read(param);
      assetsInputStream.close();
    }
    {
      InputStream assetsInputStream = assetManager.open("ncnnNet.bin");
      int available = assetsInputStream.available();
      bin = new byte[available];
      int byteCode = assetsInputStream.read(bin);
      assetsInputStream.close();
    }
    {
      InputStream assetsInputStream = assetManager.open("synset_words.txt");
      int available = assetsInputStream.available();
      words = new byte[available];
      int byteCode = assetsInputStream.read(words);
      assetsInputStream.close();
    }

    squeezencnn.Init(param, bin, words);
  }
  // Config values.
  private String inputName;
  private int inputSize;

  // Pre-allocated buffers.
  private int[] intValues;
  private float[] floatValues;
  private String[] outputNames;

  private int blockSize;

  private boolean logStats = false;

  private SqueezeNcnn detector;

  /** Initializes a native TensorFlow session for classifying images. */
  public static Classifier create(
      final AssetManager assetManager,
      SqueezeNcnn squeezencnn,

      final int inputSize,

      final int blockSize) {





    TensorFlowYoloDetector d = new TensorFlowYoloDetector();

    d.inputSize = inputSize;

    // Pre-allocate buffers.


    d.blockSize = blockSize;
    d.detector = squeezencnn;

    try
    {
      initSqueezeNcnn(assetManager,squeezencnn);//new TensorFlowInferenceInterface(assetManager, modelFilename);
    }
    catch (IOException e)
    {
      Log.e("MainActivity", "initSqueezeNcnn error");
    }


    return d;
  }

  private TensorFlowYoloDetector() {}

  private float expit(final float x) {
    return (float) (1. / (1. + Math.exp(-x)));
  }

  private void softmax(final float[] vals) {
    float max = Float.NEGATIVE_INFINITY;
    for (final float val : vals) {
      max = Math.max(max, val);
    }
    float sum = 0.0f;
    for (int i = 0; i < vals.length; ++i) {
      vals[i] = (float) Math.exp(vals[i] - max);
      sum += vals[i];
    }
    for (int i = 0; i < vals.length; ++i) {
      vals[i] = vals[i] / sum;
    }
  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {


    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.


    Trace.endSection(); // preprocessBitmap

    // Copy the input data into TensorFlow.



    // Copy the output Tensor back into the output array.
    Trace.beginSection("fetch");
    final int gridWidth = 10;//bitmap.getWidth() / blockSize;
    final int gridHeight =10;// bitmap.getHeight() / blockSize;
    final float[] output =
        new float[gridWidth * gridHeight * (NUM_CLASSES + 5) * NUM_BOXES_PER_BLOCK];

    detector.Detect(bitmap,output);
    Trace.endSection();

    // Find the best detections.
    final PriorityQueue<Recognition> pq =
        new PriorityQueue<Recognition>(
            1,
            new Comparator<Recognition>() {
              @Override
              public int compare(final Recognition lhs, final Recognition rhs) {
                // Intentionally reversed to put high confidence at the head of the queue.
                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
              }
            });

    int strid=85;
    for(int i = 0;i<gridHeight*gridWidth*NUM_BOXES_PER_BLOCK;i++){
      int offset = i*strid;
      final float xPos = output[offset+0];
      final float yPos = output[offset+1];
      final float w = output[offset+2];
      final float h = output[offset+3];
      final RectF rect =
              new RectF(
                      Math.max(0, (xPos - w / 2.f)*bitmap.getWidth()),
                      Math.max(0, (yPos - h / 2.f)*bitmap.getHeight()),
                      Math.min(bitmap.getWidth() - 1, (xPos + w / 2.f)*bitmap.getWidth()),
                      Math.min(bitmap.getHeight() - 1, (yPos + h / 2.f)*bitmap.getHeight()));

      int detectedClass = -1;
      float maxClass = 0;

      final float[] classes = new float[NUM_CLASSES];
      for (int c = 0; c < NUM_CLASSES; ++c) {
        classes[c] = output[offset + 4 + c];
      }


      for (int c = 0; c < NUM_CLASSES; ++c) {
        if (classes[c] > maxClass) {
          detectedClass = c;
          maxClass = classes[c];
        }
      }

      final float confidenceInClass = maxClass;
      if (confidenceInClass > 0.01) {

        pq.add(new Recognition("" + offset, LABELS[detectedClass], confidenceInClass, rect));
      }
    }

   /* for (int y = 0; y < gridHeight; ++y) {
      for (int x = 0; x < gridWidth; ++x) {
        for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
          final int offset =
              (gridWidth * (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5))) * y
                  + (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5)) * x
                  + (NUM_CLASSES + 5) * b;

          final float xPos = (x + expit(output[offset + 0])) *blockSize;
          final float yPos = (y + expit(output[offset + 1])) *blockSize;

          final float w = (float) (Math.exp(output[offset + 2]) * ANCHORS[2 * b + 0]) *blockSize;
          final float h = (float) (Math.exp(output[offset + 3]) * ANCHORS[2 * b + 1]) *blockSize;

          final RectF rect =
              new RectF(
                  Math.max(0, xPos - w / 2),
                  Math.max(0, yPos - h / 2),
                  Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                  Math.min(bitmap.getHeight() - 1, yPos + h / 2));
          final float confidence = expit(output[offset + 4]);

          int detectedClass = -1;
          float maxClass = 0;

          final float[] classes = new float[NUM_CLASSES];
          for (int c = 0; c < NUM_CLASSES; ++c) {
            classes[c] = output[offset + 5 + c];
          }
          softmax(classes);

          for (int c = 0; c < NUM_CLASSES; ++c) {
            if (classes[c] > maxClass) {
              detectedClass = c;
              maxClass = classes[c];
            }
          }

          final float confidenceInClass = maxClass * confidence;
          if (confidenceInClass > 0.01) {

            pq.add(new Recognition("" + offset, LABELS[detectedClass], confidenceInClass, rect));
          }
        }
    }*/

    final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
    for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
      recognitions.add(pq.poll());
    }
    Trace.endSection(); // "recognizeImage"


    return recognitions;
  }

  @Override
  public void enableStatLogging(final boolean logStats) {
    this.logStats = logStats;
  }


}
