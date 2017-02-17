using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Diagnostics;
using System.Collections.Generic;

namespace SharpLearning.Optimization.Test
{
    [TestClass]
    public class SequentialModelBasedOptimizerTest
    {
        [TestMethod]
        public void SequentialModelBasedOptimizer_OptimizeBest()
        {
            var parameters = new double[][]
            {
                new double[] { -10.0, 10.0 },
                new double[] { -10.0, 10.0 },
                new double[] { -10.0, 10.0 }
            };

            var sut = new SequentialModelBasedOptimizer(parameters, 20, 5);
            var actual = sut.OptimizeBest(Minimize);

            Assert.AreEqual(-0.78008434504774338, actual.Error, 0.0000001);
            Assert.AreEqual(3, actual.ParameterSet.Length);

            Assert.AreEqual(-7.9348252981893745, actual.ParameterSet[0], 0.0000001);
            Assert.AreEqual(0.44480908593386381, actual.ParameterSet[1], 0.0000001);
            Assert.AreEqual(0.15339395991115268, actual.ParameterSet[2], 0.0000001);
        }

        [TestMethod]
        public void SequentialModelBasedOptimizer_OptimizeBest_UsingPreviousResults()
        {
            var parameters = new double[][]
            {
                new double[] { -10.0, 10.0 },
                new double[] { -10.0, 10.0 },
                new double[] { -10.0, 10.0 }
            };

            var sut = new SequentialModelBasedOptimizer(parameters, 20, 
                PreviousParameterSets, PreviousParameterSetScores);

            var actual = sut.OptimizeBest(Minimize);

            Assert.AreEqual(actual.Error, -0.91943588723373215, 0.0000001);
            Assert.AreEqual(actual.ParameterSet.Length, 3);

            Assert.AreEqual(actual.ParameterSet[0], -7.6494288656810925, 0.0000001);
            Assert.AreEqual(actual.ParameterSet[1], 0.23936649346810235, 0.0000001);
            Assert.AreEqual(actual.ParameterSet[2], -0.034585140934172642, 0.0000001);
        }

        [TestMethod]
        public void SequentialModelBasedOptimizer_Optimize()
        {
            var parameters = new double[][] { new double[] { 0.0, 100.0 } };
            var sut = new SequentialModelBasedOptimizer(parameters, 20, 5);
            var results = sut.Optimize(Minimize2);
            var actual = new OptimizerResult[] { results.First(), results.Last() };

            var expected = new OptimizerResult[]
            {
                new OptimizerResult(new double[] { 37.713759626383109 }, 109.34382945231164),
                new OptimizerResult(new double[] { 66.810646591154239}, 34867.634010511123)
            };

            Assert.AreEqual(expected.First().Error, actual.First().Error, 0.0001);
            Assert.AreEqual(expected.First().ParameterSet.First(), actual.First().ParameterSet.First(), 0.0001);

            Assert.AreEqual(expected.Last().Error, actual.Last().Error, 0.0001);
            Assert.AreEqual(expected.Last().ParameterSet.First(), actual.Last().ParameterSet.First(), 0.0001);
        }


        OptimizerResult Minimize(double[] x)
        {
            return new OptimizerResult(x, Math.Sin(x[0]) * Math.Cos(x[1]) * (1.0 / (Math.Abs(x[2]) + 1)));
        }

        OptimizerResult Minimize2(double[] parameters)
        {
            var heights = new double[] { 1.47, 1.50, 1.52, 1.55, 1.57, 1.60, 1.63, 1.65, 1.68, 1.70, 1.73, 1.75, 1.78, 1.80, 1.83 };
            var weights = new double[] { 52.21, 53.12, 54.48, 55.84, 57.20, 58.57, 59.93, 61.29, 63.11, 64.47, 66.28, 68.10, 69.92, 72.19, 74.46 };

            var cost = 0.0;

            for (int i = 0; i < heights.Length; i++)
            {
                cost += (parameters[0] * heights[i] - weights[i]) * (parameters[0] * heights[i] - weights[i]);
            }

            return new OptimizerResult(parameters, cost);
        }


        List<double[]> PreviousParameterSets = new List<double[]>
        {
            new double[] {3.36212931823085,-7.18185403253038,-7.48963421093749},
            new double[] {0.455285520504827,-6.63131551660193,-4.74814649426758},
            new double[] {4.48816729452841,0.258455830746541,-6.52697658935887},
            new double[] {5.22501117327484,-5.30823083375964,-4.85357204212508},
            new double[] {0.112070892989669,-3.59533625356636,-2.38060772995493},
            new double[] {4.61779181455466,7.23085846046245,-6.41615969439318},
            new double[] {4.46007947885119,0.0758463233167266,-6.51458756963758},
            new double[] {4.48816729452841,0.258455830746541,-6.52697658935887},
            new double[] {4.19510496083366,10,-6.50788398840064},
            new double[] {4.02329636984216,3.38705171324023,-6.49248763970155},
            new double[] {4.19353026162903,3.55432081019386,-6.48422598826931},
            new double[] {4.34794083945136,-10,-10},
            new double[] {4.38251964409687,-7.89373555384123,-10},
            new double[] {4.35393488449434,-7.47520211015752,-10},
            new double[] {4.5011713853512,-1.41410418632009,-8.70323300070671},
            new double[] {4.5322860952287,1.85148956630954,-8.29256044884728},
            new double[] {4.53368545759782,-1.08892934485709,-6.95100625381838},
            new double[] {4.44326046857071,0.610795732721887,0.642735482421774},
            new double[] {4.46160132401457,0.632987647248452,6.38576939602329},
            new double[] {7.95099466689584,0.703323358521287,-6.69300877946318},
            new double[] {4.42169910668956,3.11335536047256,2.50501392012707},
            new double[] {4.43780512833181,10,3.1930355685055},
            new double[] {4.43241947041398,9.1179004001834,3.27005318963547},
            new double[] {4.36357679641835,0.661262658756588,-0.315184439902444},
            new double[] {4.44271866193061,-0.930237825762787,0.656993427266526},
            new double[] {4.45438655488961,0.0451136459772432,-0.282131012489168},
            new double[] {4.47043490896808,-2.221294127993,-0.80905364462375},
            new double[] {6.19172621806203,-0.513716331071088,-0.779322174275737},
            new double[] {-0.176106512613158,0.63950621447681,-0.230051809976146},
            new double[] {10,-7.56839574348162,-0.376634358952288},
            new double[] {10,-7.5562410802923,-0.490238964595083},
            new double[] {10,-3.9979204700043,-0.451491434590287},
            new double[] {10,-10,-0.277599633927654},
            new double[] {10,-10,-0.291985377880438},
            new double[] {10,-10,-0.292621110839089},
            new double[] {-8.14386771917071,-0.502862676139323,0.244721276457908},
            new double[] {-7.53428920892186,-0.199133380642047,0.545992630146148},
            new double[] {-10,-0.836690235163168,0.51740852346907},
            new double[] {-8.76359861148235,-0.430102209328472,10},
            new double[] {-8.64475216179796,-0.487405041589325,10},
            new double[] {-8.40589573151748,-0.349441409089044,10},
            new double[] {-7.02561758890906,-0.0294662352163868,0.247078719964213},
            new double[] {-6.83658100929306,0.0315501759201062,0.240148266715435},
            new double[] {-6.60670555454312,0.128941698918124,0.231554309795574},
            new double[] {3.49038917118135,-0.206632758773203,-0.15380196647166},
            new double[] {-0.0821621055898927,-0.0151014003896912,-0.246325849281556},
            new double[] {-0.0022533915639027,0.0434232901732443,-0.00155644325791027},
            new double[] {-7.95815109136047,-0.0829343774290994,0.298355424710432},
            new double[] {-7.95229891356327,0.135497208815549,0.402134135861638},
            new double[] {-5.70294815514952,-0.11644892693986,0.674653368017736},
            new double[] {-7.82579632824452,0.488643338749725,0.275171278627577},
            new double[] {-7.77622256055416,0.531772515278215,0.268041692757603},
            new double[] {-7.93482529818937,0.444809085933864,0.153393959911153},
            new double[] {-8.02966155326023,1.38970091880518,0.457092485881011},
            new double[] {-8.15592125679772,1.05911597391276,0.399791089020899},
            new double[] {-8.08773277575496,0.18545224899049,0.451556125636813},
            new double[] {-7.98754844614327,0.0277799876224925,0.270558493522855},
            new double[] {-7.72150902822271,0.339499372746043,0.330711153373899},
            new double[] {-7.99587028594922,0.690842260947225,-0.27174418554089},
            new double[] {-7.96501262426989,0.148831378604047,-10},
            new double[] {-7.97341395423525,0.148297081318717,-10},
            new double[] {-7.89735888696987,0.408029269118092,-10},
            new double[] {-7.73657022906699,-1.13805765641053,9.87528640464765},
            new double[] {-7.74467437947254,-0.953519324791359,10},
            new double[] {-7.72303844652613,-0.865877805212182,10},
        };

        List<double> PreviousParameterSetScores = new List<double>
        {
            -0.0160439475016596,
            0.0719085733865021,
            -0.125227531259282,
            -0.083550546583352,
            -0.0297342946431542,
            -0.0783377224553788,
            -0.128490688432624,
            -0.125227531259282,
            0.0971368259288108,
            0.0999252308291055,
            0.106285846916429,
            0.0712692562148148,
            0.00341824261100657,
            -0.0314802944420076,
            -0.0157250367814513,
            0.0293289531427559,
            -0.0573580442670031,
            -0.480724652822543,
            -0.105749524571127,
            0.0986754031643079,
            0.27322700484903,
            0.192614233115979,
            0.214555724711825,
            -0.563943911027358,
            -0.347643962606351,
            -0.753369044045895,
            0.325000306618909,
            -0.0447040671176907,
            -0.114285375199517,
            -0.111330536829292,
            -0.107093250697639,
            0.245575914291843,
            0.357289258107022,
            0.353310984148039,
            0.353137219821128,
            -0.674567142989078,
            -0.601925111126265,
            0.240180980923281,
            -0.0507384115617568,
            -0.0564908671109561,
            -0.0727327239786152,
            -0.541897301378144,
            -0.423592549925498,
            -0.255991195333942,
            -0.28990827202499,
            -0.0658418006628136,
            -0.00224776699325377,
            -0.763397136458902,
            -0.703248908764037,
            0.325147553093164,
            -0.692157864122856,
            -0.677663211477836,
            -0.780084345047743,
            -0.121704665850897,
            -0.333972946766054,
            -0.658688726440805,
            -0.779744348066525,
            -0.702376266710531,
            -0.599935516707696,
            -0.0893505000749157,
            -0.0892707951416537,
            -0.0833673791107632,
            -0.0382952113801735,
            -0.0523056716055908,
            -0.0584022368152105,
        };

    }
}
