package com.github.aldofsm.JavaNeuralNetworks.net;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.github.aldofsm.JavaNeuralNetworks.neuron.Input;
import com.github.aldofsm.JavaNeuralNetworks.neuron.NervousImpulseSource;
import com.github.aldofsm.JavaNeuralNetworks.neuron.Neuron;
import com.github.aldofsm.JavaNeuralNetworks.neuron.SynapticWeight;
import com.github.aldofsm.JavaNeuralNetworks.training.ErrorFunctions;
import com.github.aldofsm.JavaNeuralNetworks.training.TrainingCase;

public class FFNeuralNetwork {

	// camada de entrada
	private final Input[] inputLayer;
	// camada de saida
	private final Neuron[] outputLayer;
	// camadas ocultas
	private Set<Neuron> hiddenNeurons;
	// funções de erro para cada unidade da camada de saida
	private ErrorFunctions[] errorFunctions;
	// fator de randomização dos pesos
	private double weightRF = 0.1;

	public FFNeuralNetwork(int numberInputs, Neuron... outputNeurons) {
		inputLayer = new Input[numberInputs];
		outputLayer = outputNeurons;
		errorFunctions = new ErrorFunctions[outputNeurons.length];
		hiddenNeurons = new HashSet<Neuron>();
		for (int i = 0; i < numberInputs; i++) {
			inputLayer[i] = new Input(0);
		}
		for (int i = 0; i < outputNeurons.length; i++) {
			errorFunctions[i] = ErrorFunctions.SQUARED_ERROR;
		}
	}

	public void addSynapse(NervousImpulseSource source, Neuron receptor) {
		if (source instanceof Neuron && !hiddenNeurons.contains(source))
			hiddenNeurons.add((Neuron) source);
		if (!hiddenNeurons.contains(receptor))
			hiddenNeurons.add(receptor);
		receptor.addInputSynapse(source, new SynapticWeight((2 * Math.random() - 1) * weightRF));
	}

	public void removeSynapse(NervousImpulseSource source, Neuron receptor) {
		receptor.removeInputSynapse(source);
	}

	public void setErroFunction(int outputIndex, ErrorFunctions function) {
		errorFunctions[outputIndex] = function;
	}

	public void addNeuron(Neuron neuron) {
		hiddenNeurons.add(neuron);
	}

	public boolean removeNeuron(Neuron neuron) {
		neuron.removeAllSynapses();
		return hiddenNeurons.remove(neuron);
	}

	private void setInputValues(double[] values) {
		for (int i = 0; i < inputLayer.length; i++) {
			inputLayer[i].setValue(values[i]);
		}
	}

	public Input getInput(int index) {
		return inputLayer[index];
	}

	private void resetVisited() {
		for (Neuron neuron : hiddenNeurons) {
			neuron.setVisited(false);
		}
		for (Neuron neuron : outputLayer) {
			neuron.setVisited(false);
		}
	}

	public void train(List<TrainingCase> examples, int numberEpochs) {

	}

	public List<Double> getOutputs(double... inputs) {
		setInputValues(inputs);
		resetVisited();
		List<Double> outputs = new ArrayList<Double>();
		for (Neuron neuron : outputLayer) {
			outputs.add(neuron.getOutput());
		}
		return outputs;
	}

}
