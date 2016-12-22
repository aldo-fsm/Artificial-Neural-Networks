package com.github.aldofsm.JavaNeuralNetworks.tests;

import com.github.aldofsm.JavaNeuralNetworks.training.DataSet;

public class DataSetTest {
	public static void main(String[] args) {

		DataSet ds = new DataSet(3, 2);

		ds.addTrainingCase(1, 2, 3, 4, 5);
		ds.addTrainingCase(4, -2, -3, 34, 15);
		ds.addTrainingCase(31, 32, 33, 34, 35);
		ds.addTrainingCase(41, 42, 3, 4, 5);
		ds.addTrainingCase(41, 42, 34, 44, 45);
		// ds.addTrainingCase(1,32,33,43,5);

		ds.setMiniBatchSize(2);
		System.out.println(ds.getInputMatrix(0));
		System.out.println(ds.getInputMatrix(1));
		System.out.println(ds.getInputMatrix(2));

		System.out.println(ds.getOutputMatrix(0));
		System.out.println(ds.getOutputMatrix(1));
		System.out.println(ds.getOutputMatrix(2));
	}
}
