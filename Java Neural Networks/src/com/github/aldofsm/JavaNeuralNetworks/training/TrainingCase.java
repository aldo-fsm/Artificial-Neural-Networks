package com.github.aldofsm.JavaNeuralNetworks.training;

import java.util.Arrays;

public class TrainingCase {

	private double[] inputs;
	private double[] desiredOutputs;

	public TrainingCase(int numberInputs, int numberOutputs, double... values) {

		inputs = new double[numberInputs];
		desiredOutputs = new double[numberOutputs];

		for (int i = 0; i < values.length; i++) {
			if (i < numberInputs)
				inputs[i] = values[i];
			else
				desiredOutputs[i - numberInputs] = values[i];
		}

	}

	public TrainingCase(double[] inputs, double[] outputs) {
		this.inputs = inputs;
		this.desiredOutputs = outputs;
	}

	public double[] getInputs() {
		return inputs;
	}

	public double[] getDesiredOutput() {
		return desiredOutputs;
	}

	@Override
	public String toString() {
		return Arrays.toString(inputs) + " -> " + Arrays.toString(desiredOutputs);
	}
}
