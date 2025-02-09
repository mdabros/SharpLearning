using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.XGBoost.Models;

namespace SharpLearning.XGBoost.Test.Models;

[TestClass]
public class FeatureImportanceParserTest
{
    readonly double m_delta = 0.0000001;

    [TestMethod]
    public void FeatureImportanceParser_Parse()
    {
        var actual = FeatureImportanceParser.ParseFromTreeDump(m_textTrees, 9);

        var expected = new double[]
        {
            29.94274,
            0,
            624.692,
            0,
            12.9966,
            43.939899999999994,
            134.3104,
            0,
            19.014670000000002
        };

        Assert.AreEqual(expected.Length, actual.Length);

        for (var i = 0; i < expected.Length; i++)
        {
            Assert.AreEqual(expected[i], actual[i], m_delta);
        }
    }

    readonly string[] m_textTrees = [m_tree1, m_tree2];

    const string m_tree1 = @"booster[0]
0:[f2<2.695] yes=1,no=2,missing=1,gain=343.922,cover=214
	1:[f6<9.81] yes=3,no=4,missing=3,gain=74.1261,cover=61
		3:[f8<0.13] yes=7,no=8,missing=7,gain=10.7401,cover=37
			7:leaf=0.404167,cover=35
			8:leaf=0.1,cover=2
		4:[f4<72.205] yes=9,no=10,missing=9,gain=12.9966,cover=24
			9:leaf=0.0444444,cover=8
			10:leaf=0.205882,cover=16
	2:[f5<1.28] yes=5,no=6,missing=5,gain=23.4366,cover=153
		5:[f0<1.51715] yes=11,no=12,missing=11,gain=10.462,cover=151
			11:leaf=0.0527778,cover=53
			12:leaf=-0.0020202,cover=98
		6:leaf=0.3,cover=2";


    const string m_tree2 = @"booster[1]
0:[f2<2.695] yes=1,no=2,missing=1,gain=280.77,cover=214
	1:[f6<9.81] yes=3,no=4,missing=3,gain=60.1843,cover=61
		3:[f8<0.13] yes=7,no=8,missing=7,gain=8.27457,cover=37
			7:leaf=0.364873,cover=35
			8:leaf=0.0933333,cover=2
		4:[f0<1.52174] yes=9,no=10,missing=9,gain=10.9722,cover=24
			9:leaf=0.19627,cover=14
			10:leaf=0.0537255,cover=10
	2:[f5<1.28] yes=5,no=6,missing=5,gain=20.5033,cover=153
		5:[f0<1.51715] yes=11,no=12,missing=11,gain=8.50854,cover=151
			11:leaf=0.0475977,cover=53
			12:leaf=-0.00182022,cover=98
		6:leaf=0.28,cover=2";
}
