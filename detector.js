import * as tf from "@tensorflow/tfjs-node";
import fs from "fs/promises";
import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import { readdirSync, readFileSync } from "fs";

(async function main() {
  try {
    // cocoDetect()
    mobileNetPredict();
  } catch (error) {
    console.log(error.message);
  }
})();

// works with pre-trained coco model (80 recognizable classes)
async function cocoDetect() {
  const images = readdirSync("./images");
  const model = await cocoSsd.load();

  images.forEach(async (img) => {
    const currentImageBuffer = await fs.readFile(`./images/${img}`);
    const tensor = tf.node.decodeImage(currentImageBuffer);
    const currentDetection = await model.detect(tensor);
    try {
      console.log(
        `There is most likely a ${currentDetection[0].class} in ${img}`
      );
    } catch (ex) {
      console.log(`We couldn't find a prediction for ${img}`);
    }
  });
}

//works with re-trained mobileNet_v2 (1000 recognizable classes)
async function mobileNetPredict() {
  const model = await tf.loadLayersModel("file://colabModel__TFJSM/model.json");
  const classnames = getClassNames();

  const images = readdirSync("./images");

  images.forEach(async (img) => {
    const imageBuffer = await fs.readFile(`./images/${img}`);
    const imageTensor = tf.node.decodeImage(new Uint8Array(imageBuffer));
    const resizedImage = tf.image.resizeBilinear(imageTensor, [224, 224]);
    const preprocessedImage = resizedImage.toFloat().div(255.0).expandDims(0); // Normalize and add batch dimension

    const predictions = model.predict(preprocessedImage);
    // Prints Tensor itself
    // console.log(predictions);
    const predictionsArray = await predictions.array();

    // Find the highest probability
    const highestProbability = Math.max(...predictionsArray[0]);
    // Find the index of the class with the highest probability (argmax)
    const argmaxIndex = predictionsArray[0].indexOf(
      Math.max(...predictionsArray[0])
    );

    console.log(`Prediction for ${img}`);
    console.log(
      `${classnames[argmaxIndex].className}, probability: ${highestProbability} \n`
    );
  });
}

function getClassNames() {
  const filePath = "./colabModel__TFJSM/LOC_synset_mapping.txt";
  const classLabels = [];
  const lines = readFileSync(filePath, "utf8").split("\n");
  lines.forEach((line) => {
    const parts = line.trim().split(" ");
    const classId = parts[0];
    const className = parts.slice(1).join(" ");
    const classLabel = { classId, className };
    classLabels.push(classLabel);
  });
  return classLabels;
}
