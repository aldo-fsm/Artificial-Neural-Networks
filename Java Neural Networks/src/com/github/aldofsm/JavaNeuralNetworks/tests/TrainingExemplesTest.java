package com.github.aldofsm.JavaNeuralNetworks.tests;

import java.lang.ref.Reference;

import com.github.aldofsm.JavaNeuralNetworks.training.TrainingCase;

public class TrainingExemplesTest {
	public static void main(String[] args) {
		TrainingCase t1 = new TrainingCase(3, 2, 1, 2, 3, 4, 5);
		TrainingCase t2 = new TrainingCase(new double[] { 1, 2, 3 }, new double[] { 4, 5});
		System.out.println(t1);
		System.out.println(t2);
	}
}
