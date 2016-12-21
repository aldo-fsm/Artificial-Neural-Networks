package com.github.aldofsm.JavaNeuralNetworks.neuron;

public class Input implements NervousImpulseSource {

	private double value = 0;

	public Input(double value) {
		this.value = value;
	}

	@Override
	public double getOutput() {
		return value;
	}

	public void setValue(double value) {
		this.value = value;
	}

}
