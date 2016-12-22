package com.github.aldofsm.JavaNeuralNetworks.neuron;

import java.util.HashMap;

public abstract class Neuron implements NervousImpulseSource {

	protected boolean visited = false;

	// derivada parcial do erro em relação a entrada total no neuronio
	protected double error = Double.NaN;
	protected double learningRate;
	protected double currentOutput;

	// input synapses
	protected HashMap<NervousImpulseSource, SynapticWeight> dendrites;
	// output synapses
	protected HashMap<Neuron, SynapticWeight> axon;

	public Neuron() {
		dendrites = new HashMap<NervousImpulseSource, SynapticWeight>();
		axon = new HashMap<Neuron, SynapticWeight>();
		addInputSynapse(Input.ONE, new SynapticWeight(0));
	}

	public void addInputSynapse(NervousImpulseSource source, SynapticWeight weight) {
		dendrites.put(source, weight);
		if (source instanceof Neuron) {
			((Neuron) source).axon.put(this, weight);
		}
	}

	public void addOutputSynapse(Neuron receptor, SynapticWeight weight) {
		axon.put(receptor, weight);
		receptor.dendrites.put(this, weight);
	}

	public void removeInputSynapse(NervousImpulseSource source) {
		dendrites.remove(source);
		if (source instanceof Neuron) {
			((Neuron) source).removeOutputSynapse(this);
		}
	}

	public void removeAllSynapses() {
		for (NervousImpulseSource source : dendrites.keySet()) {
			removeInputSynapse(source);
		}
		for (Neuron receptor : axon.keySet()) {
			removeOutputSynapse(receptor);
		}
	}

	public void removeOutputSynapse(Neuron receptor) {
		axon.remove(receptor);
		receptor.removeInputSynapse(this);
	}

	public abstract void calculateOutput();

	public abstract void calculateError();

	public abstract void adjustWeights();

	public boolean isVisited() {
		return visited;
	}

	public void setVisited(boolean visited) {
		this.visited = visited;
	}

	public double getError() {
		return error;
	}

	public void setError(double error) {
		this.error = error;
	}

	public double getLearningRate() {
		return learningRate;
	}

	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

}
