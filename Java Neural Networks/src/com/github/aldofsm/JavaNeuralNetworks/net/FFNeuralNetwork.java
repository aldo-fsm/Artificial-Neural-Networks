package com.github.aldofsm.JavaNeuralNetworks.net;

import org.ejml.simple.SimpleMatrix;

/**
 * Rede Neural Artificial feedfoward multicamada. Todas unidades utilizam a
 * função sigmoide logistica como função de ativação.
 * 
 * @author aldo
 * 
 */
public class FFNeuralNetwork {

	private double learningRate;
	private double weightRandomFactor = 0.1;

	private SimpleMatrix[] layersOutputs;
	private SimpleMatrix[] errors;
	private SimpleMatrix[] weights;
	private SimpleMatrix[] bias;

	public FFNeuralNetwork(int numberInputs, int numberHiddenLayers, int numberOutputs) {

		layersOutputs = new SimpleMatrix[numberHiddenLayers + 2];

		bias = new SimpleMatrix[numberHiddenLayers + 1];
		weights = new SimpleMatrix[numberHiddenLayers + 1];
		errors = new SimpleMatrix[numberHiddenLayers + 1];

	}

}
