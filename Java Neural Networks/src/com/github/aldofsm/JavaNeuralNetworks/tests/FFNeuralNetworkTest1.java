package com.github.aldofsm.JavaNeuralNetworks.tests;

import com.github.aldofsm.JavaNeuralNetworks.net.FFNeuralNetwork;
import com.github.aldofsm.JavaNeuralNetworks.training.DataSet;

public class FFNeuralNetworkTest1 {
	public static void main(String[] args) {

//		FFNeuralNetwork ffnn = new FFNeuralNetwork(2, 1, 1);
//		DataSet data = new DataSet(2, 1);
//		data.addTrainingCase(1, 1, 1);
//		data.addTrainingCase(1, 0, 0);
//		data.addTrainingCase(0, 0, 0);
//		data.setMiniBatchSize(3);
//		ffnn.train(data, 1000);
//		System.out.println(ffnn.output(1, 1));
//		System.out.println(ffnn.output(1, 0));
//		System.out.println(ffnn.output(0, 0));
//		System.out.println(ffnn.output(0, 1));

		FFNeuralNetwork ffnn = new FFNeuralNetwork(3, 3, 5);
		DataSet data = new DataSet(3, 3);
		data.addTrainingCase(0,0,1,0,1,0);
		data.addTrainingCase(0,1,0,0,1,1);
		data.addTrainingCase(0,1,1,1,0,0);
		data.addTrainingCase(1,0,0,1,0,1);

		data.setMiniBatchSize(4);
		ffnn.setWeightRandomAmplitude(1);
		
		ffnn.train(data, 1000);
		System.out.println(ffnn.output(0,0,1));
		System.out.println(ffnn.output(0,1,0));
		System.out.println(ffnn.output(0,1,1));
		System.out.println(ffnn.output(1,0,0));
	}
}
