namespace SharpLearning.XGBoost.FSharp.Test.Learners

module FSharpLearner =

    open System
    open System.IO
    open System.Linq
    open Microsoft.VisualStudio.TestTools.UnitTesting
    open SharpLearning.XGBoost
    open SharpLearning.InputOutput.Csv
    open SharpLearning.Metrics.Regression
    open SharpLearning.XGBoost.Learners
    open SharpLearning.XGBoost.FSharp.Test

    let CreateLearner ()=
        RegressionXGBoostLearner(
            maximumTreeDepth = 3,
            learningRate = 0.1,
            estimators = 100,
            silent = true,
            objective = RegressionObjective.LinearRegression,
            boosterType = BoosterType.GBTree,
            treeMethod = TreeMethod.Auto,
            numberOfThreads = -1,
            gamma = 0.0,
            minChildWeight = 1,
            maxDeltaStep = 0,
            subSample = 1.0,
            colSampleByTree = 1.0,
            colSampleByLevel = 1.0,
            l1Regularization = 0.0,
            l2Reguralization = 1.0,
            scalePosWeight = 1.0,
            baseScore = 0.5,
            seed = 0,
            missing= Double.NaN)
  

    [<TestClass>]
    type RegressionXGBoostLearnerTest() =

       let m_delta = 0.0000001

       [<TestMethod>]
       member this.RegressionXGBoostLearnerFSharp_Learn () = 
            let parser = new CsvParser(fun () -> new StringReader(Resources.Glass) :> TextReader)
            let observations = parser.EnumerateRows(fun v -> v <> "Target").ToF64Matrix()
            let targets = parser.EnumerateRows("Target").ToF64Vector()

            let sut = CreateLearner()

            using (sut.Learn(observations, targets)) (fun model ->

                let predictions = model.Predict(observations)

                let evaluator = new MeanSquaredErrorRegressionMetric()
                let error = evaluator.Error(targets, predictions)

                Assert.AreEqual(0.0795934933096642, error, m_delta)
            )

        
        [<TestMethod>]
        member this.RegressionXGBoostLearnerFSharp_Learn_Indexed () =
            let parser = new CsvParser(fun () -> new StringReader(Resources.Glass) :> TextReader)
            let observations = parser.EnumerateRows(fun v -> v <> "Target").ToF64Matrix()
            let targets = parser.EnumerateRows("Target").ToF64Vector()

            let shuffle (rng:Random) (org:_[]) = 
                let arr = Array.copy org
                let max = (arr.Length - 1)
                let randomSwap (arr:_[]) i =
                    let pos = rng.Next(max)
                    let tmp = arr.[pos]
                    arr.[pos] <- arr.[i]
                    arr.[i] <- tmp
                    arr
   
                [|0..max|] |> Array.fold randomSwap arr

            let indices = Enumerable.Range(0, targets.Length)
                           .ToArray()
                           |>shuffle (new Random(42))
                           |>Array.take ((int)((float)targets.Length * 0.7))

            let sut = CreateLearner()

 
            using (sut.Learn(observations, targets, indices)) (fun model ->

                let predictions = model.Predict(observations)

                let evaluator = new MeanSquaredErrorRegressionMetric()
                let error = evaluator.Error(targets, predictions)

                Assert.AreEqual(0.361022826612673, error, m_delta) 
            )
