using System;
using System.Collections.Generic;
using System.Text;

namespace SharpLearning.Backend
{
    class TrainingApi
    {
        interface IDataSource
        {
            Tuple<ITensorSymbol, ITensorSymbol> NextBatch(int size); // Return type should probably be different.
        }

        // training API brainstorm
        interface IOptimizer
        {
            ITensorSymbol Minimize(ITensorSymbol cost);
        };

        interface ILoss
        {
            ITensorSymbol Loss(ITensorSymbol targets, ITensorSymbol predictions);
        }

        void Suggestion1()
        {
            // This will probably not possible with current CNTK and TensorFlowSharp C# API state.
            // Might also increase back and forth copying between CPU/GPU.

            // var graph = DefineGraph(int[] shape, IBackend backend)
            // var loss = new CrossEntropy();
            // var metric = new ClassificationAccuracy();
            // var optimizer = new SGD(learningRate: 0.001);

            //using (var session = StartSession(graph))            
            // for epochs

            //    reset/shuffle datasource

            //    datasource prefetcher

            //    for iterations
            //          Tuple<ITensorSymbol> batch = source.NextBatch(batchSize: 64); // load next minibatch
            //          ITensorSymbol predictions = session.Forward(graph, batch); // forward mini batch

            //          ITensorSymbol cost = loss(batch.Targets, predictions); // calculate loss
            //          ITensorSymbol accuracy = metric(batch.Targets, predictions); // optional
            //          ITensorSymbol gradients = session.Backward(graph, batch); // back probabagate gradients
            //          optimizer.Update(graph, gradients) // update parameters

            //    Trace.WriteLine($"Epoch {e}: Cost {cost.GetValue()}, Metric {metric.GetValue()}")

            // done trace result
            // save model graph.Save()

        }

        void Suggestion2()
        {
            // This is probably more realistic with current CNTK and TensorFlowSharp C# API state,
            // However, requires the implementation of an optimizer in TensorFlowSharp. This concept exist in the python API, so can be ported.

            // var graph = DefineGraph(int[] shape, IBackend backend)
            // var loss = new CrossEntropy();
            // var metric = new ClassificationError().Error(placeholder, graph(placeholder))
            // var optimizer = new SGD(learningRate: 0.001).Minimize(loss: loss);

            //using (var session = StartSession(graph))            
            // for epochs

            //    reset/shuffle datasource

            //    datasource prefetcher

            //    for iterations
            //          Tuple<ITensorSymbol> batch = source.NextBatch(batchSize: 64); // load next minibatch
            //          ITensorSymbol cost = session.Run(optimizer, batch); // forward, backward, update parameters, return cost
            //          ITensorSymbol accuracy = metric(batch)
            
            //    Trace.WriteLine($"Epoch {e}: Cost {cost.GetValue()}, Metric {metric.GetValue()}")

            // done trace result
            // save model graph.Save()
        }
    }
}
