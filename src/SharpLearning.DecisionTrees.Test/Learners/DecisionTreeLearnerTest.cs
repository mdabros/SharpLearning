using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DecisionTrees.Learners;
using System;

namespace SharpLearning.DecisionTrees.Test.Learners
{
    [TestClass]
    public class DecisionTreeLearnerTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void DecisionTreeLearner_TreeBuilderIsNull()
        {
            new DecisionTreeLearner(null);
        }
    }
}
