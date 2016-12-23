package com.github.aldofsm.JavaNeuralNetworks.training;

import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.ejml.simple.SimpleMatrix;

public class DataSet {

	private final int numberInputs;
	private final int numberOutputs;
	private int miniBatchSize;
	private int numberMiniBatches;

	private List<Double[]> inputs;
	private List<Double[]> desiredOutputs;

	public DataSet(int numberInputs, int numberOutputs) {
		this.numberInputs = numberInputs;
		this.numberOutputs = numberOutputs;

		inputs = new ArrayList<Double[]>();
		desiredOutputs = new ArrayList<Double[]>();

		setMiniBatchSize(1);

	}

	public void addTrainingCase(double... values) {
		if (values.length != numberInputs + numberOutputs)
			throw new InvalidParameterException("Numero de parametros invalido");
		Double[] inputs = new Double[numberInputs];
		Double[] outputs = new Double[numberOutputs];
		for (int i = 0; i < values.length; i++) {
			if (i < numberInputs)
				inputs[i] = values[i];
			else
				outputs[i - numberInputs] = values[i - numberInputs];
		}
		addTrainingCase(inputs, outputs);
	}

	public void addTrainingCase(Double[] inputs, Double[] outputs) {
		if (inputs.length != numberInputs || outputs.length != numberOutputs)
			throw new InvalidParameterException("numero de entradas ou saidas invalido");
		this.inputs.add(inputs);
		this.desiredOutputs.add(outputs);
	}

	private SimpleMatrix partition(List<Double[]> list, int start, int end, int numRow) {
		List<Double[]> aux = list.subList(start, end);
		SimpleMatrix matrix = new SimpleMatrix(numRow, aux.size());
		for (int i = 0; i < numRow; i++) {
			for (int j = 0; j < aux.size(); j++) {
				matrix.set(i, j, aux.get(j)[i]);
			}
		}
		return matrix;
	}

	public List<SimpleMatrix> inputMatrixList() {
		List<SimpleMatrix> list = new ArrayList<SimpleMatrix>();
		for (int i = 0; i < numberMiniBatches; i++) {
			list.add(getInputMatrix(i));
		}
		return list;
	}

	public List<SimpleMatrix> outputMatrixList() {
		List<SimpleMatrix> list = new ArrayList<SimpleMatrix>();
		for (int i = 0; i < numberMiniBatches; i++) {
			list.add(getOutputMatrix(i));
		}
		return list;
	}

	private SimpleMatrix getInputMatrix(int partitionIndex) {
		int start = miniBatchSize * partitionIndex;
		int end = start + miniBatchSize;
		if (end > inputs.size())
			end = inputs.size();
		return partition(inputs, start, end, numberInputs);
	}

	private SimpleMatrix getOutputMatrix(int partitionIndex) {
		int start = miniBatchSize * partitionIndex;
		int end = start + miniBatchSize;
		if (end > desiredOutputs.size())
			end = desiredOutputs.size();
		return partition(desiredOutputs, start, end, numberOutputs);
	}

	public int getMiniBatchSize() {
		return miniBatchSize;
	}

	public void setMiniBatchSize(int miniBatchSize) {
		this.miniBatchSize = miniBatchSize;
		numberMiniBatches = inputs.size() / miniBatchSize;
		if (inputs.size() % miniBatchSize != 0)
			numberMiniBatches++;
	}

}
