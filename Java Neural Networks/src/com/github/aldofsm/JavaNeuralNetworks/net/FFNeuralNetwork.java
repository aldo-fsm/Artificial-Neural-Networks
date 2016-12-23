package com.github.aldofsm.JavaNeuralNetworks.net;

import java.util.List;
import java.util.Random;
import java.util.function.Function;

import org.ejml.simple.SimpleMatrix;
import org.xml.sax.InputSource;

import com.github.aldofsm.JavaNeuralNetworks.training.DataSet;

/**
 * Rede Neural Artificial feedfoward multicamada. Todas unidades utilizam a
 * função sigmoide logistica como função de ativação.
 * 
 */
public class FFNeuralNetwork {

	private double learningRate;
	private double weightRandomAmplitude = 0.1;

	private SimpleMatrix inputLayer;
	private SimpleMatrix[] hiddenLayers;
	private SimpleMatrix outputLayer;
	private SimpleMatrix[] errors;
	private SimpleMatrix[] weights;
	private SimpleMatrix[] bias;
	private final int inSize;
	private final int outSize;
	private final int numHidden;

	private static final Function<Double, Double> SIGMOID = x -> 1 / (1 + Math.exp(-x));

	public FFNeuralNetwork(int numberInputs, int numberOutputs, int... hiddenLayerSizes) {
		numHidden = hiddenLayerSizes.length;
		inSize = numberInputs;
		outSize = numberOutputs;

		hiddenLayers = new SimpleMatrix[hiddenLayerSizes.length];

		bias = new SimpleMatrix[numHidden + 1];
		weights = new SimpleMatrix[numHidden + 1];
		errors = new SimpleMatrix[numHidden + 1];
		Random random = new Random();
		int i;
		for (i = 0; i < numHidden; i++) {
			int numRows = i != 0 ? hiddenLayerSizes[i - 1] : numberInputs;
			int numCols = i < numHidden ? hiddenLayerSizes[i] : numberOutputs;
			bias[i] = new SimpleMatrix(hiddenLayerSizes[i], 1);
			weights[i] = SimpleMatrix.random(numRows, numCols, -weightRandomAmplitude, weightRandomAmplitude, random);
		}
		bias[numHidden] = new SimpleMatrix(numberOutputs, 1);
		weights[numHidden] = SimpleMatrix.random(hiddenLayerSizes[i - 1], numberOutputs, -weightRandomAmplitude,
				weightRandomAmplitude, random);

	}

	private void forwardPropagation() {
		int batchSize = inputLayer.numRows();

		SimpleMatrix totalInput = weights[0].transpose().mult(inputLayer);
		totalInput = totalInput.plus(repMat(bias[0], 1, batchSize));
		hiddenLayers[0] = applyFuncion(SIGMOID, totalInput);

		int i;
		for (i = 1; i < hiddenLayers.length; i++) {
			totalInput = weights[i].transpose().mult(hiddenLayers[i - 1]);
			totalInput = totalInput.plus(repMat(bias[i], 1, batchSize));
			hiddenLayers[i] = applyFuncion(SIGMOID, totalInput);
		}

		totalInput = weights[weights.length - 1].transpose().mult(hiddenLayers[i]);
		totalInput = totalInput.plus(repMat(bias[i], 1, batchSize));
		outputLayer = applyFuncion(SIGMOID, totalInput);
	}

	private void backPropagation() {

		for (int i = errors.length - 2; i >= 1; i--) {
			errors[i] = weights[i + 1].mult(errors[i + 1]);
			errors[i] = errors[i].elementMult(applyFuncion(x -> x * (1 - x), hiddenLayers[i - 1]));
		}
		errors[0] = weights[1].mult(errors[1]);
		errors[0] = errors[0].elementMult(applyFuncion(x -> x * (1 - x), inputLayer));

	}

	public void train(DataSet trainingData, int epochs) {
		List<SimpleMatrix> trainingInputs = trainingData.inputMatrixList();
		List<SimpleMatrix> trainingOutputs = trainingData.outputMatrixList();
		for (int epoch = 1; epoch <= epochs; epoch++) {
			for (int i = 0; i < trainingInputs.size(); i++) {
				inputLayer = trainingInputs.get(i);
				forwardPropagation();
				SimpleMatrix outputError = outputLayer.minus(trainingOutputs.get(i));
				outputError = outputError.elementMult(applyFuncion(x -> x * (1 - x), outputLayer));
				errors[errors.length - 1] = outputError;
				backPropagation();
			}
		}
	}

	private static SimpleMatrix applyFuncion(Function<Double, Double> function, SimpleMatrix matrix) {
		double element;
		SimpleMatrix result = new SimpleMatrix(matrix);
		for (int i = 0; i < matrix.numRows(); i++) {
			for (int j = 0; j < matrix.numCols(); j++) {
				element = matrix.get(i, j);
				result.set(i, j, function.apply(element));
			}
		}
		return result;
	}

	private static SimpleMatrix repMat(SimpleMatrix m, int rows, int cols) {
		SimpleMatrix result = new SimpleMatrix(rows * m.numRows(), cols * m.numCols());
		for (int i = 0; i < result.numRows(); i += m.numRows()) {
			for (int j = 0; j < result.numCols(); j += m.numCols()) {
				result = result.combine(i, j, m);
			}
		}
		return result;
	}

}
