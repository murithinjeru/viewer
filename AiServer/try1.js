import * as tf from '@tensorflow/tfjs';
import cornerstone from 'cornerstone-core';
import cornerstoneTools from 'cornerstone-tools';
import OHIF from '@ohif/core';

import * as tf from '@tensorflow/tfjs';
import setCornerstoneLayout from './utils/setCornerstoneLayout.js';
import { getEnabledElement } from './state';
import CornerstoneViewportDownloadForm from './CornerstoneViewportDownloadForm';
function loadMobilenet() {
  return tf.loadModel('models/Converted/model.json');
}
function loadImage(src) {
  return new Promise((resolve, reject) => {
    let img = cornerstone.getEnabledElements()[0].element;
    img.src = src;
    img.onload = () => resolve(tf.fromPixels(img));
    img.onerror = err => reject(err);
  });
}
function cropImage(img) {
  const width = img.shape[0];  const height = img.shape[1];
  // use the shorter side as the size to which we will crop
  const shorterSide = Math.min(img.shape[0], img.shape[1]);
  // calculate beginning and ending crop points
  const startingHeight = (height - shorterSide) / 2;
  const startingWidth = (width - shorterSide) / 2;
  const endingHeight = startingHeight + shorterSide;
  const endingWidth = startingWidth + shorterSide;
  // return image data cropped to those points
  return img.slice([startingWidth, startingHeight, 0], [endingWidth, endingHeight, 3]);
  }
function resizeImage(image) {
  return tf.image.resizeBilinear(image, [224, 224]);
}
function batchImage(image) {
  // Expand our tensor to have an additional dimension, whose size is 1
  const batchedImage = image.expandDims(0);
  // Turn pixel data into a float between -1 and 1.
  return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
}
function loadAndProcessImage(image) {
  const croppedImage = cropImage(image);
  const resizedImage = resizeImage(croppedImage);
  const batchedImage = batchImage(resizedImage);
  return batchedImage;
}

import drum from './data/pretrained-model-data/drum.jpg';
loadMobilenet().then(pretrainedModel => {
  loadImage(drum).then(img => {
    const processedImage = loadAndProcessImage(img);
    const prediction = pretrainedModel.predict(processedImage);
    // Because of the way Tensorflow.js works, you must call print on a Tensor instead of console.log.
  prediction.print();
  prediction.as1D().argMax().print();
  });
});
