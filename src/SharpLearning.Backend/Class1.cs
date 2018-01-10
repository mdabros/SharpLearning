using System;
using System.Collections.Generic;
using System.Text;

namespace SharpLearning.Backend
{
    class Class1
    {
        const int SurfaceViewRows = 512;
        const int SurfaceViewCols = 512;
        const int SurfaceViewChannels = 3;
        const string SurfaceViewChannelOrder = "NCHW"; //N=Batch C=Channels H=Height W=Width
                                                       //NHWC(TensorFlow default) and NCHW(cuDNN default)


        void test()
        {
            using (var backend = new CntkBackend(defaultDevice))
            //using (var backend = new TensorFlowBackend(defaultDevice))
            // Create data source
            //IDataSource source = CreateSource();

            // Define graph
            //IGraph test = DefineGraph(int inputSize);

            // Session
            //using (var session = StartSession(graph))
            //using (var session = StartSession(function))
            {
                // for epochs

                //    reset/shuffle datasource

                //    datasource prefetcher

                //    for iterations

                //        train batch

                //    trace progress

                // done trace result

                // save model graph.Save()
            }
        }
    }


    interface IGraph
    {
    }

    interface ISymbol { }

    interface ITensorSymbol { }

    interface IParametersTensorSymbol { }

    interface IOperatorSymbol { }

    IGraph DefineGraph(int[] shape, IBackend backend)
    {
        using (var graph = backend.CreateGraph())
        {
            IPlaceholderSymbol placeholder = graph.Placeholder(asdada);

            IParametersTensorSymbol w = graph.Parameters(shape, init);
            IParametersTensorSymbol b = graph.Parameters(shape, init);

            graph.Operators.Dense(w, b)

            IParametersTensorSymbol convolutionParameters = graph.Parameters();

            var convOperator = graph.Convolution(w, b);


            graph.Operators.ReLU(convOperator, shape)

            

            var pooling = graph.Operators.Pooling(convOperator, shape, "average");


        }

        using (var graph = backend.CreateLayers())
        {
            graph.Layers().
            layers.
            }
        return null;
    }
}
