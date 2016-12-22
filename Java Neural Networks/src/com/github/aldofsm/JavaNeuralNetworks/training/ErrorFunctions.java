package com.github.aldofsm.JavaNeuralNetworks.training;

import java.util.function.Function;

public enum ErrorFunctions {

	SQUARED_ERROR((x) -> 0.5 * Math.pow(x[0] - x[1], 2), (x) -> x[0] - x[1])

	;

	Function<Double[], Double> function;
	Function<Double[], Double> derivative;

	private ErrorFunctions(Function<Double[], Double> function, Function<Double[], Double> derivative) {
		this.function = function;
		this.derivative = derivative;
	}

	public double apply(double output, double desiredOutput) {
		return function.apply(new Double[] { output, desiredOutput });
	}

	public double applyDerivative(double output, double desiredOutput) {
		return derivative.apply(new Double[] { output, desiredOutput });
	}

}
