// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
PoseNet
The original PoseNet model was ported to TensorFlow.js by Dan Oved.
*/

import EventEmitter from "events";
import * as tf from "@tensorflow/tfjs";
import callCallback from "../utils/callcallback";

import * as mpPose from '@mediapipe/pose';
import * as posedetection from '@tensorflow-models/pose-detection';


const DEFAULTS = {
  detectionType: "single", // 'multiple'
};

class PoseDetector extends EventEmitter {
  /**
   * @typedef {Object} options
   * @property {string} architecture - default 'MobileNetV1',
   * @property {number} inputResolution - default 257,
   * @property {number} outputStride - default 16
   * @property {boolean} flipHorizontal - default false
   * @property {number} minConfidence - default 0.5
   * @property {number} maxPoseDetections - default 5
   * @property {number} scoreThreshold - default 0.5
   * @property {number} nmsRadius - default 20
   * @property {String} detectionType - default single
   * @property {number} nmsRadius - default 0.75,
   * @property {number} quantBytes - default 2,
   * @property {string} modelUrl - default null
   */
  /**
   * Create a PoseNet model.
   * @param {HTMLVideoElement || p5.Video} video  - Optional. A HTML video element or a p5 video element.
   * @param {options} options - Optional. An object describing a model accuracy and performance.
   * @param {String} detectionType - Optional. A String value to run 'single' or 'multiple' estimation.
   * @param {function} callback  Optional. A function to run once the model has been loaded.
   *    If no callback is provided, it will return a promise that will be resolved once the
   *    model has loaded.
   */
  constructor(video, options, detectionType, callback) {
    super();
    this.video = video;
    /**
     * The type of detection. 'single' or 'multiple'
     * @type {String}
     * @public
     */

    this.detectionType = detectionType || options.detectionType || DEFAULTS.detectionType;
    this.ready = callCallback(this.load(), callback);
    this.then = this.ready.then;
  }

  async load() {

    this.net = await posedetection.createDetector(posedetection.SupportedModels.BlazePose, {
      runtime: "mediapipe",
      modelType: "heavy",
      solutionPath: `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${mpPose.VERSION}`,
    });

    if (this.video) {
      if (this.video.readyState === 0) {
        await new Promise(resolve => {
          this.video.onloadeddata = () => resolve();
        });
      }
      if (this.detectionType === "single") {
        this.singlePose();
      } else {
        this.multiPose();
      }
    }
    return this;
  }




  getInput(inputOr) {
    let input;
    if (
      inputOr instanceof HTMLImageElement ||
      inputOr instanceof HTMLVideoElement ||
      inputOr instanceof HTMLCanvasElement ||
      inputOr instanceof ImageData
    ) {
      input = inputOr;
    } else if (
      typeof inputOr === "object" &&
      (inputOr.elt instanceof HTMLImageElement ||
        inputOr.elt instanceof HTMLVideoElement ||
        inputOr.elt instanceof ImageData)
    ) {
      input = inputOr.elt; // Handle p5.js image and video
    } else if (typeof inputOr === "object" && inputOr.canvas instanceof HTMLCanvasElement) {
      input = inputOr.canvas; // Handle p5.js image
    } else {
      input = this.video;
    }

    return input;
  }

  /**
   * Given an image or video, returns an array of objects containing pose estimations
   *    using single or multi-pose detection.
   * @param {HTMLVideoElement || p5.Video || function} inputOr
   * @param {function} cb
   */
  /* eslint max-len: ["error", { "code": 180 }] */
  async singlePose(inputOr, cb) {
    const input = this.getInput(inputOr);

    const result = await this.net.estimatePoses(input, {
      maxPoses: 1,
      flipHorizontal: false,
    });
    // const poseWithParts = this.mapParts(pose);
    // const result = [{ pose: poseWithParts, skeleton: this.skeleton(pose.keypoints) }];
    this.emit("pose", result);

    if (this.video) {
      return tf.nextFrame().then(() => this.singlePose());
    }

    if (typeof cb === "function") {
      cb(result);
    }

    return result;
  }

  /**
   * Given an image or video, returns an array of objects containing pose
   *    estimations using single or multi-pose detection.
   * @param {HTMLVideoElement || p5.Video || function} inputOr
   * @param {function} cb
   */
  async multiPose(inputOr, cb) {
    const input = this.getInput(inputOr);

    const result = await this.net.estimatePoses(input, {
      maxPoses: 1,
      flipHorizontal: false,
    });

    // const posesWithParts = poses.map(pose => this.mapParts(pose));
    // const result = posesWithParts.map(pose => ({ pose, skeleton: this.skeleton(pose.keypoints) }));
    this.emit("pose", result);
    if (this.video) {
      return tf.nextFrame().then(() => this.multiPose());
    }

    if (typeof cb === "function") {
      cb(result);
    }

    return result;
  }
}

const poseDetector = (videoOrOptionsOrCallback, optionsOrCallback, cb) => {
  let video;
  let options = {};
  let callback = cb;
  let detectionType = null;

  if (videoOrOptionsOrCallback instanceof HTMLVideoElement) {
    video = videoOrOptionsOrCallback;
  } else if (
    typeof videoOrOptionsOrCallback === "object" &&
    videoOrOptionsOrCallback.elt instanceof HTMLVideoElement
  ) {
    video = videoOrOptionsOrCallback.elt; // Handle a p5.js video element
  } else if (typeof videoOrOptionsOrCallback === "object") {
    options = videoOrOptionsOrCallback;
  } else if (typeof videoOrOptionsOrCallback === "function") {
    callback = videoOrOptionsOrCallback;
  }

  if (typeof optionsOrCallback === "object") {
    options = optionsOrCallback;
  } else if (typeof optionsOrCallback === "string") {
    detectionType = optionsOrCallback;
  }

  if (typeof optionsOrCallback === "function") {
    callback = optionsOrCallback;
  }

  return new PoseDetector(video, options, detectionType, callback);
};

export default poseDetector;
