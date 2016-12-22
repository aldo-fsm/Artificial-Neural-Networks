package com.github.aldofsm.JavaNeuralNetworks.tests;

import com.github.aldofsm.JavaNeuralNetworks.net.FFNeuralNetwork;
import com.github.aldofsm.JavaNeuralNetworks.neuron.Neuron;
import com.github.aldofsm.JavaNeuralNetworks.neuron.SigmoidNeuron;

public class FFNeuralNetworkTest1 {
	public static void main(String[] args) {

		Neuron n1 = new SigmoidNeuron();
		Neuron n2 = new SigmoidNeuron();
		Neuron n3 = new SigmoidNeuron();
		Neuron n4 = new SigmoidNeuron();
		Neuron n5 = new SigmoidNeuron();

		FFNeuralNetwork net = new FFNeuralNetwork(3, n1, n2);

		net.addSynapse(net.getInput(0), n3);
		net.addSynapse(net.getInput(1), n3);
		net.addSynapse(net.getInput(0), n4);
		net.addSynapse(net.getInput(1), n4);
		net.addSynapse(n4, n5);
		net.addSynapse(n5, n1);
		net.addSynapse(n3, n2);

		System.out.println(net.getOutputs(1, 2, 3));
	}
}
