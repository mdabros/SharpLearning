﻿using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Optimization.ParameterSamplers;

namespace SharpLearning.Optimization.Test.ParameterSamplers;

[TestClass]
public class RandomUniformTest
{
    [TestMethod]
    public void RandomUniform_Sample_Continous()
    {
        var sut = new RandomUniform(32);

        var actual = new double[10];
        for (var i = 0; i < actual.Length; i++)
        {
            actual[i] = sut.Sample(min: 20, max: 200, parameterType: ParameterType.Continuous);
        }

        var expected = new double[] { 99.8935983236384, 57.2098020451189, 44.4149092419142, 89.9002946307418, 137.643828772774, 114.250629522954, 63.8914499915631, 109.294177409864, 188.567149950455, 33.2731248034505 };
        Assert.AreEqual(expected.Length, actual.Length);
        for (var i = 0; i < expected.Length; i++)
        {
            Assert.AreEqual(expected[i], actual[i], 0.000001);
        }
    }

    public static void RandomUniformIntergers_Sample_Integer()
    {
        var sut = new RandomUniform(32);

        var actual = new double[10];
        for (var i = 0; i < actual.Length; i++)
        {
            actual[i] = sut.Sample(min: 20, max: 200, parameterType: ParameterType.Discrete);
        }

        var expected = new double[] { 100, 57, 44, 90, 138, 114, 64, 109, 189, 33 };
        Assert.AreEqual(expected.Length, actual.Length);
        for (var i = 0; i < expected.Length; i++)
        {
            Assert.AreEqual(expected[i], actual[i], 0.000001);
        }
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void RandomUniform_Throw_On_Min_Larger_Than_Max()
    {
        var sut = new RandomUniform(32);
        sut.Sample(min: 20, max: 10, parameterType: ParameterType.Continuous);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void RandomUniform_Throw_On_Min_Equals_Than_Max()
    {
        var sut = new RandomUniform(32);
        sut.Sample(min: 20, max: 20, parameterType: ParameterType.Continuous);
    }
}
