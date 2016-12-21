package com.github.aldofsm.JavaNeuralNetworks.neuron;

import java.util.HashMap;

public abstract class Neuron implements NervousImpulseSource {

	protected boolean visited = false;

	// derivada parcial do erro em relação a entrada total no neuronio
	protected double error = Double.NaN;
	protected double learningRate;

	// input synapses
	private HashMap<NervousImpulseSource, SynapticWeight> dendrites = new HashMap<NervousImpulseSource, SynapticWeight>();
	// output synapses
	private HashMap<Neuron, SynapticWeight> axon = new HashMap<Neuron, SynapticWeight>();

	public void addInputSynapse(NervousImpulseSource source, SynapticWeight weight) {
		dendrites.put(source, weight);
		if (source instanceof Neuron) {
			((Neuron) source).addOutputSynapse(this, weight);
		}
	}

	public void addOutputSynapse(Neuron receptor, SynapticWeight weight) {

	}

	protected abstract void calculateError();

	public abstract void adjustWeights();

}
