using System.Collections.Generic;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Extensions;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.DecisionTrees.Test.Models
{
    [TestClass]
    public class RegressionDecisionTreeModelTest
    {
        readonly string m_regressionDecisionTreeModelString = "<?xml version=\"1.0\" encoding=\"utf-16\"?>\r\n<RegressionDecisionTreeModel xmlns:i=\"http://www.w3.org/2001/XMLSchema-instance\" z:Id=\"1\" xmlns:z=\"http://schemas.microsoft.com/2003/10/Serialization/\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Models\">\r\n  <Tree xmlns:d2p1=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\" z:Id=\"2\">\r\n    <d2p1:Nodes z:Id=\"3\" z:Size=\"25\">\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>-1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>0</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>10</d2p1:RightIndex>\r\n        <d2p1:Value>0.397254</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>-1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>2</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>1</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>5</d2p1:RightIndex>\r\n        <d2p1:Value>0.20301550000000002</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>-1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>3</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>2</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>4</d2p1:RightIndex>\r\n        <d2p1:Value>0.14998250000000002</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>0</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>3</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>-0.054810500000000005</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>4</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>0.07189454545454545</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>-1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>6</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>5</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>7</d2p1:RightIndex>\r\n        <d2p1:Value>0.3190235</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>2</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>6</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>1.094141117647059</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>-1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>8</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>7</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>9</d2p1:RightIndex>\r\n        <d2p1:Value>0.3652945</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>3</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>8</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>0.8030192857142858</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>4</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>9</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>1.1078694999999998</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>-1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>11</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>10</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>18</d2p1:RightIndex>\r\n        <d2p1:Value>0.5957425000000001</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>-1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>12</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>11</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>13</d2p1:RightIndex>\r\n        <d2p1:Value>0.48716800000000005</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>5</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>12</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>1.88108975</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>-1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>14</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>13</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>15</d2p1:RightIndex>\r\n        <d2p1:Value>0.5380695</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>6</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>14</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>2.084306555555555</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>-1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>16</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>15</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>17</d2p1:RightIndex>\r\n        <d2p1:Value>0.5625644999999999</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>7</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>16</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>1.81453875</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>8</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>17</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>2.072091</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>-1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>19</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>18</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>24</d2p1:RightIndex>\r\n        <d2p1:Value>0.8071625</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>-1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>20</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>19</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>21</d2p1:RightIndex>\r\n        <d2p1:Value>0.6214955</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>9</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>20</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>3.1442664000000002</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>-1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>22</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>21</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>23</d2p1:RightIndex>\r\n        <d2p1:Value>0.6617200000000001</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>10</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>22</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>2.8252029999999997</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>11</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>23</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>2.983283454545455</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>12</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>24</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>3.9871632</d2p1:Value>\r\n      </d2p1:Node>\r\n    </d2p1:Nodes>\r\n    <d2p1:Probabilities xmlns:d3p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"4\" z:Size=\"13\">\r\n      <d3p1:ArrayOfdouble z:Id=\"5\" z:Size=\"0\" />\r\n      <d3p1:ArrayOfdouble z:Id=\"6\" z:Size=\"0\" />\r\n      <d3p1:ArrayOfdouble z:Id=\"7\" z:Size=\"0\" />\r\n      <d3p1:ArrayOfdouble z:Id=\"8\" z:Size=\"0\" />\r\n      <d3p1:ArrayOfdouble z:Id=\"9\" z:Size=\"0\" />\r\n      <d3p1:ArrayOfdouble z:Id=\"10\" z:Size=\"0\" />\r\n      <d3p1:ArrayOfdouble z:Id=\"11\" z:Size=\"0\" />\r\n      <d3p1:ArrayOfdouble z:Id=\"12\" z:Size=\"0\" />\r\n      <d3p1:ArrayOfdouble z:Id=\"13\" z:Size=\"0\" />\r\n      <d3p1:ArrayOfdouble z:Id=\"14\" z:Size=\"0\" />\r\n      <d3p1:ArrayOfdouble z:Id=\"15\" z:Size=\"0\" />\r\n      <d3p1:ArrayOfdouble z:Id=\"16\" z:Size=\"0\" />\r\n      <d3p1:ArrayOfdouble z:Id=\"17\" z:Size=\"0\" />\r\n    </d2p1:Probabilities>\r\n    <d2p1:TargetNames xmlns:d3p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"18\" z:Size=\"200\">\r\n      <d3p1:double>1.88318</d3p1:double>\r\n      <d3p1:double>0.063908</d3p1:double>\r\n      <d3p1:double>3.042257</d3p1:double>\r\n      <d3p1:double>2.305004</d3p1:double>\r\n      <d3p1:double>-0.067698</d3p1:double>\r\n      <d3p1:double>1.662809</d3p1:double>\r\n      <d3p1:double>3.275749</d3p1:double>\r\n      <d3p1:double>1.118077</d3p1:double>\r\n      <d3p1:double>2.095059</d3p1:double>\r\n      <d3p1:double>1.181912</d3p1:double>\r\n      <d3p1:double>0.221663</d3p1:double>\r\n      <d3p1:double>0.938453</d3p1:double>\r\n      <d3p1:double>4.149409</d3p1:double>\r\n      <d3p1:double>3.105444</d3p1:double>\r\n      <d3p1:double>1.896278</d3p1:double>\r\n      <d3p1:double>-0.121345</d3p1:double>\r\n      <d3p1:double>3.161652</d3p1:double>\r\n      <d3p1:double>4.135358</d3p1:double>\r\n      <d3p1:double>0.859063</d3p1:double>\r\n      <d3p1:double>1.170272</d3p1:double>\r\n      <d3p1:double>1.68796</d3p1:double>\r\n      <d3p1:double>1.979745</d3p1:double>\r\n      <d3p1:double>0.06869</d3p1:double>\r\n      <d3p1:double>4.052137</d3p1:double>\r\n      <d3p1:double>3.156316</d3p1:double>\r\n      <d3p1:double>2.95063</d3p1:double>\r\n      <d3p1:double>0.068935</d3p1:double>\r\n      <d3p1:double>2.85402</d3p1:double>\r\n      <d3p1:double>0.999743</d3p1:double>\r\n      <d3p1:double>4.048082</d3p1:double>\r\n      <d3p1:double>0.230923</d3p1:double>\r\n      <d3p1:double>0.816693</d3p1:double>\r\n      <d3p1:double>0.130713</d3p1:double>\r\n      <d3p1:double>-0.537706</d3p1:double>\r\n      <d3p1:double>-0.339109</d3p1:double>\r\n      <d3p1:double>2.124538</d3p1:double>\r\n      <d3p1:double>2.708292</d3p1:double>\r\n      <d3p1:double>4.01739</d3p1:double>\r\n      <d3p1:double>4.004021</d3p1:double>\r\n      <d3p1:double>1.022555</d3p1:double>\r\n      <d3p1:double>3.657442</d3p1:double>\r\n      <d3p1:double>4.073619</d3p1:double>\r\n      <d3p1:double>0.011994</d3p1:double>\r\n      <d3p1:double>3.640429</d3p1:double>\r\n      <d3p1:double>1.808497</d3p1:double>\r\n      <d3p1:double>1.431404</d3p1:double>\r\n      <d3p1:double>3.935544</d3p1:double>\r\n      <d3p1:double>1.162152</d3p1:double>\r\n      <d3p1:double>-0.22733</d3p1:double>\r\n      <d3p1:double>-0.068728</d3p1:double>\r\n      <d3p1:double>0.825051</d3p1:double>\r\n      <d3p1:double>2.008645</d3p1:double>\r\n      <d3p1:double>0.664566</d3p1:double>\r\n      <d3p1:double>4.180202</d3p1:double>\r\n      <d3p1:double>0.864845</d3p1:double>\r\n      <d3p1:double>1.851174</d3p1:double>\r\n      <d3p1:double>2.761993</d3p1:double>\r\n      <d3p1:double>4.075914</d3p1:double>\r\n      <d3p1:double>0.110229</d3p1:double>\r\n      <d3p1:double>2.181987</d3p1:double>\r\n      <d3p1:double>3.152528</d3p1:double>\r\n      <d3p1:double>3.046564</d3p1:double>\r\n      <d3p1:double>0.128954</d3p1:double>\r\n      <d3p1:double>1.062726</d3p1:double>\r\n      <d3p1:double>3.651742</d3p1:double>\r\n      <d3p1:double>4.40195</d3p1:double>\r\n      <d3p1:double>3.022888</d3p1:double>\r\n      <d3p1:double>2.874917</d3p1:double>\r\n      <d3p1:double>2.946801</d3p1:double>\r\n      <d3p1:double>2.893644</d3p1:double>\r\n      <d3p1:double>0.072131</d3p1:double>\r\n      <d3p1:double>1.748275</d3p1:double>\r\n      <d3p1:double>1.912047</d3p1:double>\r\n      <d3p1:double>3.710686</d3p1:double>\r\n      <d3p1:double>1.719148</d3p1:double>\r\n      <d3p1:double>2.17409</d3p1:double>\r\n      <d3p1:double>3.656357</d3p1:double>\r\n      <d3p1:double>3.522504</d3p1:double>\r\n      <d3p1:double>2.234126</d3p1:double>\r\n      <d3p1:double>1.859772</d3p1:double>\r\n      <d3p1:double>2.097017</d3p1:double>\r\n      <d3p1:double>0.001794</d3p1:double>\r\n      <d3p1:double>1.231928</d3p1:double>\r\n      <d3p1:double>2.953862</d3p1:double>\r\n      <d3p1:double>-0.116803</d3p1:double>\r\n      <d3p1:double>2.638864</d3p1:double>\r\n      <d3p1:double>3.943428</d3p1:double>\r\n      <d3p1:double>-0.328513</d3p1:double>\r\n      <d3p1:double>-0.099866</d3p1:double>\r\n      <d3p1:double>-0.030836</d3p1:double>\r\n      <d3p1:double>2.359786</d3p1:double>\r\n      <d3p1:double>3.212581</d3p1:double>\r\n      <d3p1:double>0.188975</d3p1:double>\r\n      <d3p1:double>1.904109</d3p1:double>\r\n      <d3p1:double>3.007114</d3p1:double>\r\n      <d3p1:double>3.845834</d3p1:double>\r\n      <d3p1:double>3.079411</d3p1:double>\r\n      <d3p1:double>1.939739</d3p1:double>\r\n      <d3p1:double>2.880078</d3p1:double>\r\n      <d3p1:double>3.063577</d3p1:double>\r\n      <d3p1:double>4.116296</d3p1:double>\r\n      <d3p1:double>-0.240963</d3p1:double>\r\n      <d3p1:double>4.066299</d3p1:double>\r\n      <d3p1:double>4.011834</d3p1:double>\r\n      <d3p1:double>0.07771</d3p1:double>\r\n      <d3p1:double>3.103069</d3p1:double>\r\n      <d3p1:double>2.811897</d3p1:double>\r\n      <d3p1:double>-0.10463</d3p1:double>\r\n      <d3p1:double>0.025216</d3p1:double>\r\n      <d3p1:double>4.330063</d3p1:double>\r\n      <d3p1:double>3.087091</d3p1:double>\r\n      <d3p1:double>2.269988</d3p1:double>\r\n      <d3p1:double>4.010701</d3p1:double>\r\n      <d3p1:double>3.119542</d3p1:double>\r\n      <d3p1:double>3.723411</d3p1:double>\r\n      <d3p1:double>2.792078</d3p1:double>\r\n      <d3p1:double>2.192787</d3p1:double>\r\n      <d3p1:double>2.081305</d3p1:double>\r\n      <d3p1:double>1.714463</d3p1:double>\r\n      <d3p1:double>0.885854</d3p1:double>\r\n      <d3p1:double>1.028187</d3p1:double>\r\n      <d3p1:double>1.951497</d3p1:double>\r\n      <d3p1:double>1.709427</d3p1:double>\r\n      <d3p1:double>0.144068</d3p1:double>\r\n      <d3p1:double>3.88024</d3p1:double>\r\n      <d3p1:double>0.921876</d3p1:double>\r\n      <d3p1:double>1.979316</d3p1:double>\r\n      <d3p1:double>3.085768</d3p1:double>\r\n      <d3p1:double>3.476122</d3p1:double>\r\n      <d3p1:double>3.993679</d3p1:double>\r\n      <d3p1:double>3.07788</d3p1:double>\r\n      <d3p1:double>-0.105365</d3p1:double>\r\n      <d3p1:double>-0.164703</d3p1:double>\r\n      <d3p1:double>1.096814</d3p1:double>\r\n      <d3p1:double>3.092879</d3p1:double>\r\n      <d3p1:double>2.987926</d3p1:double>\r\n      <d3p1:double>2.061264</d3p1:double>\r\n      <d3p1:double>2.746854</d3p1:double>\r\n      <d3p1:double>0.71671</d3p1:double>\r\n      <d3p1:double>0.103831</d3p1:double>\r\n      <d3p1:double>0.023776</d3p1:double>\r\n      <d3p1:double>-0.033299</d3p1:double>\r\n      <d3p1:double>1.942286</d3p1:double>\r\n      <d3p1:double>-0.006338</d3p1:double>\r\n      <d3p1:double>3.808753</d3p1:double>\r\n      <d3p1:double>0.652799</d3p1:double>\r\n      <d3p1:double>4.053747</d3p1:double>\r\n      <d3p1:double>4.56929</d3p1:double>\r\n      <d3p1:double>-0.032773</d3p1:double>\r\n      <d3p1:double>2.066236</d3p1:double>\r\n      <d3p1:double>0.222785</d3p1:double>\r\n      <d3p1:double>1.089268</d3p1:double>\r\n      <d3p1:double>1.487788</d3p1:double>\r\n      <d3p1:double>2.852033</d3p1:double>\r\n      <d3p1:double>0.024486</d3p1:double>\r\n      <d3p1:double>3.73775</d3p1:double>\r\n      <d3p1:double>0.045017</d3p1:double>\r\n      <d3p1:double>0.001238</d3p1:double>\r\n      <d3p1:double>3.892763</d3p1:double>\r\n      <d3p1:double>2.819376</d3p1:double>\r\n      <d3p1:double>2.830665</d3p1:double>\r\n      <d3p1:double>0.234633</d3p1:double>\r\n      <d3p1:double>1.810782</d3p1:double>\r\n      <d3p1:double>4.237235</d3p1:double>\r\n      <d3p1:double>3.034768</d3p1:double>\r\n      <d3p1:double>1.742106</d3p1:double>\r\n      <d3p1:double>1.16925</d3p1:double>\r\n      <d3p1:double>0.831165</d3p1:double>\r\n      <d3p1:double>3.729376</d3p1:double>\r\n      <d3p1:double>1.823205</d3p1:double>\r\n      <d3p1:double>4.02197</d3p1:double>\r\n      <d3p1:double>1.262939</d3p1:double>\r\n      <d3p1:double>4.159518</d3p1:double>\r\n      <d3p1:double>2.039114</d3p1:double>\r\n      <d3p1:double>4.101837</d3p1:double>\r\n      <d3p1:double>2.778672</d3p1:double>\r\n      <d3p1:double>1.228284</d3p1:double>\r\n      <d3p1:double>1.73662</d3p1:double>\r\n      <d3p1:double>-0.195046</d3p1:double>\r\n      <d3p1:double>-0.063215</d3p1:double>\r\n      <d3p1:double>3.305268</d3p1:double>\r\n      <d3p1:double>2.063627</d3p1:double>\r\n      <d3p1:double>0.89884</d3p1:double>\r\n      <d3p1:double>2.701692</d3p1:double>\r\n      <d3p1:double>1.992909</d3p1:double>\r\n      <d3p1:double>3.811393</d3p1:double>\r\n      <d3p1:double>4.353857</d3p1:double>\r\n      <d3p1:double>2.635641</d3p1:double>\r\n      <d3p1:double>2.856311</d3p1:double>\r\n      <d3p1:double>1.352682</d3p1:double>\r\n      <d3p1:double>2.336459</d3p1:double>\r\n      <d3p1:double>2.111651</d3p1:double>\r\n      <d3p1:double>0.121726</d3p1:double>\r\n      <d3p1:double>3.264605</d3p1:double>\r\n      <d3p1:double>2.103446</d3p1:double>\r\n      <d3p1:double>0.896855</d3p1:double>\r\n      <d3p1:double>4.22085</d3p1:double>\r\n      <d3p1:double>-0.217283</d3p1:double>\r\n      <d3p1:double>-0.300577</d3p1:double>\r\n      <d3p1:double>0.006014</d3p1:double>\r\n    </d2p1:TargetNames>\r\n    <d2p1:VariableImportance xmlns:d3p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"19\" z:Size=\"2\">\r\n      <d3p1:double>0</d3p1:double>\r\n      <d3p1:double>364.5635685044051</d3p1:double>\r\n    </d2p1:VariableImportance>\r\n  </Tree>\r\n  <m_variableImportance xmlns:d2p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Ref=\"19\" i:nil=\"true\" />\r\n</RegressionDecisionTreeModel>";

        [TestMethod]
        public void RegressionDecisionTreeModel_Predict_Single()
        {
            var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

            var learner = new RegressionDecisionTreeLearner(100, 4, 2, 0.1, 42);
            var sut = learner.Learn(observations, targets);

            var rows = targets.Length;
            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = sut.Predict(observations.Row(i));
            }

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.032120286249559482, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionDecisionTreeModel_Predict_Multiple()
        {
            var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

            var learner = new RegressionDecisionTreeLearner(100, 4, 2, 0.1, 42);
            var sut = learner.Learn(observations, targets);

            var predictions = sut.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.032120286249559482, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionDecisionTreeModel_Predict_Multiple_Indexed()
        {
            var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

            var learner = new RegressionDecisionTreeLearner(100, 4, 2, 0.1, 42);
            var sut = learner.Learn(observations, targets);

            var indices = new int[] { 0, 3, 4, 5, 6, 7, 8, 9, 20, 21 };
            var predictions = sut.Predict(observations, indices);

            var indexedTargets = targets.GetIndices(indices);
            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(indexedTargets, predictions);

            Assert.AreEqual(0.023821615502626264, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionDecisionTreeModel_GetVariableImportance()
        {
            var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

            var featureNameToIndex = new Dictionary<string, int> { { "F1", 0 }, { "F2", 1 } };

            var learner = new RegressionDecisionTreeLearner(100, 4, 2, 0.1, 42);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetVariableImportance(featureNameToIndex);
            var expected = new Dictionary<string, double> { { "F2", 100.0 }, { "F1", 0.0 } };

            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void RegressionDecisionTreeModel_GetRawVariableImportance()
        {
            var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

            var featureNameToIndex = new Dictionary<string, int> { { "F1", 0 }, { "F2", 1 } };

            var learner = new RegressionDecisionTreeLearner(100, 4, 2, 0.1, 42);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetRawVariableImportance();
            var expected = new double[] { 0.0, 364.56356850440511 };

            Assert.AreEqual(expected.Length, actual.Length);

            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.000001);
            }
        }

        [TestMethod]
        public void RegressionDecisionTreeModel_Save()
        {
            var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

            var learner = new RegressionDecisionTreeLearner(100, 4, 2, 0.1, 42);
            var sut = learner.Learn(observations, targets);

            var writer = new StringWriter();
            sut.Save(() => writer);

            var actual = writer.ToString();
            Assert.AreEqual(m_regressionDecisionTreeModelString, actual);
        }

        [TestMethod]
        public void RegressionDecisionTreeModel_Load()
        {
            var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

            var reader = new StringReader(m_regressionDecisionTreeModelString);
            var sut = RegressionDecisionTreeModel.Load(() => reader);

            var predictions = sut.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.032120286249559482, error, 0.0000001);
        }
    }
}
