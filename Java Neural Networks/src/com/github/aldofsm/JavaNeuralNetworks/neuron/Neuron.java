package com.github.aldofsm.JavaNeuralNetworks.neuron;

import java.util.HashMap;

public abstract class Neuron implements NervousImpulseSource {

	protected boolean visited = false;

	// derivada parcial do erro em relação a entrada total no neuronio
	protected double error = Double.NaN;
	protected double learningRate;
	protected double currentOutput;

	// input synapses
	protected HashMap<NervousImpulseSource, SynapticWeight> dendrites = new HashMap<NervousImpulseSource, SynapticWeight>();
	// output synapses
	protected HashMap<Neuron, SynapticWeight> axon = new HashMap<Neuron, SynapticWeight>();

	public void addInputSynapse(NervousImpulseSource source, SynapticWeight weight) {
		dendrites.put(source, weight);
		if (source instanceof Neuron) {
			((Neuron) source).addOutputSynapse(this, weight);
		}
	}

	public void addOutputSynapse(Neuron receptor, SynapticWeight weight) {
		axon.put(receptor, weight);
		receptor.addInputSynapse(this, weight);
	}

	public abstract void calculateOutput();

	public abstract void calculateError();

	public abstract void adjustWeights();

}
