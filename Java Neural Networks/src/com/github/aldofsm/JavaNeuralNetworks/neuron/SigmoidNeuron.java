package com.github.aldofsm.JavaNeuralNetworks.neuron;

public class SigmoidNeuron extends Neuron {

	private static double sigmoid(double input) {
		return 1 / (1 + Math.exp(-input));
	}

	@Override
	public double getOutput() {
		if (!visited)
			calculateOutput();
		return currentOutput;
	}

	@Override
	public void adjustWeights() {
		SynapticWeight weight;
		for (NervousImpulseSource source : dendrites.keySet()) {
			weight = dendrites.get(source);
			weight.add(-learningRate * error * source.getOutput());
		}
	}

	@Override
	public void calculateError() {
		double error = 0;
		for (Neuron receptor : axon.keySet()) {
			error += receptor.error * axon.get(receptor).getValue();
		}
		this.error = currentOutput * (1 - currentOutput) * error;
	}

	@Override
	public void calculateOutput() {
		double input = 0;
		// somatorio ponderado
		for (NervousImpulseSource source : dendrites.keySet()) {
			input += source.getOutput() * dendrites.get(source).getValue();
		}
		visited = true;
		currentOutput = sigmoid(input);
	}

}
