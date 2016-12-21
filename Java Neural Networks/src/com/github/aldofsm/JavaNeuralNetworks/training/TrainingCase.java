package com.github.aldofsm.JavaNeuralNetworks.training;

import java.util.ArrayList;
import java.util.List;

public class TrainingCase {

	private List<Double> inputs;
	private List<Double> desiredOutputs;

	public TrainingCase(int numberInputs, int numberOutputs, double... values) {

		inputs = new ArrayList<Double>();
		desiredOutputs = new ArrayList<Double>();

		for (int i = 0; i < values.length; i++) {
			if (i < numberInputs)
				inputs.add(values[i]);
			else
				desiredOutputs.add(values[i]);
		}

	}

	public TrainingCase(double[] inputs, double[] outputs) {
		this.inputs = new ArrayList<Double>();
		this.desiredOutputs = new ArrayList<Double>();

		for (Double value : inputs) {
			this.inputs.add(value);
		}

		for (Double value : outputs) {
			this.desiredOutputs.add(value);
		}
	}

	public List<Double> getInputs() {
		return inputs;
	}

	public List<Double> getDesiredOutput() {
		return desiredOutputs;
	}

	@Override
	public String toString() {
		return inputs + " -> " + desiredOutputs;
	}
}
