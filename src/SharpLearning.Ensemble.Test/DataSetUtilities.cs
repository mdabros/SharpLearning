﻿using System.IO;
using SharpLearning.Containers.Matrices;
using SharpLearning.InputOutput.Csv;

namespace SharpLearning.Ensemble.Test;

public static class DataSetUtilities
{
    public static (F64Matrix observations, double[] targets) LoadAptitudeDataSet()
    {
        var parser = new CsvParser(() => new StringReader(AptitudeData));
        var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
        var targets = parser.EnumerateRows("Pass").ToF64Vector();
        return (observations, targets);
    }

    public static (F64Matrix observations, double[] targets) LoadGlassDataSet()
    {
        var parser = new CsvParser(() => new StringReader(GlassData));
        var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
        var targets = parser.EnumerateRows("Target").ToF64Vector();
        return (observations, targets);
    }

    public static (F64Matrix observations, double[] targets) LoadDecisionTreeDataSet()
    {
        var parser = new CsvParser(() => new StringReader(DecisionTreeData));
        var observations = parser.EnumerateRows(v => v != "T").ToF64Matrix();
        var targets = parser.EnumerateRows("T").ToF64Vector();
        return (observations, targets);
    }

    public const string AptitudeData =
@"AptitudeTestScore;PreviousExperience_month;Pass
5;6;0
1;15;0
1;12;0
4;6;0
1;15;1
1;6;0
4;16;1
1;10;1
3;12;0
4;26;1
5;2;1
1;12;0
3;18;0
3;3;0
1;24;1
2;8;0
1;9;0
4;18;0
4;22;1
5;3;1
4;12;0
4;24;1
2;18;1
2;6;0
1;8;0
5;12;0";

    public const string DecisionTreeData =
@"F1;F2;T
1;0.409175;1.88318
1;0.182603;0.063908
1;0.663687;3.042257
1;0.517395;2.305004
1;0.013643;-0.067698
1;0.469643;1.662809
1;0.725426;3.275749
1;0.39435;1.118077
1;0.50776;2.095059
1;0.237395;1.181912
1;0.057534;0.221663
1;0.36982;0.938453
1;0.976819;4.149409
1;0.616051;3.105444
1;0.4137;1.896278
1;0.105279;-0.121345
1;0.670273;3.161652
1;0.952758;4.135358
1;0.272316;0.859063
1;0.303697;1.170272
1;0.486698;1.68796
1;0.51181;1.979745
1;0.195865;0.06869
1;0.986769;4.052137
1;0.785623;3.156316
1;0.797583;2.95063
1;0.081306;0.068935
1;0.659753;2.85402
1;0.37527;0.999743
1;0.819136;4.048082
1;0.142432;0.230923
1;0.215112;0.816693
1;0.04127;0.130713
1;0.044136;-0.537706
1;0.131337;-0.339109
1;0.463444;2.124538
1;0.671905;2.708292
1;0.946559;4.01739
1;0.904176;4.004021
1;0.306674;1.022555
1;0.819006;3.657442
1;0.845472;4.073619
1;0.156258;0.011994
1;0.857185;3.640429
1;0.400158;1.808497
1;0.375395;1.431404
1;0.885807;3.935544
1;0.23996;1.162152
1;0.14864;-0.22733
1;0.143143;-0.068728
1;0.321582;0.825051
1;0.509393;2.008645
1;0.355891;0.664566
1;0.938633;4.180202
1;0.348057;0.864845
1;0.438898;1.851174
1;0.781419;2.761993
1;0.911333;4.075914
1;0.032469;0.110229
1;0.499985;2.181987
1;0.771663;3.152528
1;0.670361;3.046564
1;0.176202;0.128954
1;0.39217;1.062726
1;0.911188;3.651742
1;0.872288;4.40195
1;0.733107;3.022888
1;0.610239;2.874917
1;0.732739;2.946801
1;0.714825;2.893644
1;0.076386;0.072131
1;0.559009;1.748275
1;0.427258;1.912047
1;0.841875;3.710686
1;0.558918;1.719148
1;0.533241;2.17409
1;0.956665;3.656357
1;0.620393;3.522504
1;0.56612;2.234126
1;0.523258;1.859772
1;0.476884;2.097017
1;0.176408;0.001794
1;0.303094;1.231928
1;0.609731;2.953862
1;0.017774;-0.116803
1;0.622616;2.638864
1;0.886539;3.943428
1;0.148654;-0.328513
1;0.10435;-0.099866
1;0.116868;-0.030836
1;0.516514;2.359786
1;0.664896;3.212581
1;0.004327;0.188975
1;0.425559;1.904109
1;0.743671;3.007114
1;0.935185;3.845834
1;0.6973;3.079411
1;0.444551;1.939739
1;0.683753;2.880078
1;0.755993;3.063577
1;0.90269;4.116296
1;0.094491;-0.240963
1;0.873831;4.066299
1;0.99181;4.011834
1;0.185611;0.07771
1;0.694551;3.103069
1;0.657275;2.811897
1;0.118746;-0.10463
1;0.084302;0.025216
1;0.945341;4.330063
1;0.785827;3.087091
1;0.530933;2.269988
1;0.879594;4.010701
1;0.65277;3.119542
1;0.879338;3.723411
1;0.764739;2.792078
1;0.504884;2.192787
1;0.554203;2.081305
1;0.493209;1.714463
1;0.363783;0.885854
1;0.316465;1.028187
1;0.580283;1.951497
1;0.542898;1.709427
1;0.112661;0.144068
1;0.816742;3.88024
1;0.234175;0.921876
1;0.402804;1.979316
1;0.709423;3.085768
1;0.867298;3.476122
1;0.993392;3.993679
1;0.71158;3.07788
1;0.133643;-0.105365
1;0.052031;-0.164703
1;0.366806;1.096814
1;0.697521;3.092879
1;0.787262;2.987926
1;0.47671;2.061264
1;0.721417;2.746854
1;0.230376;0.71671
1;0.104397;0.103831
1;0.197834;0.023776
1;0.129291;-0.033299
1;0.528528;1.942286
1;0.009493;-0.006338
1;0.998533;3.808753
1;0.363522;0.652799
1;0.901386;4.053747
1;0.832693;4.56929
1;0.119002;-0.032773
1;0.487638;2.066236
1;0.153667;0.222785
1;0.238619;1.089268
1;0.208197;1.487788
1;0.750921;2.852033
1;0.183403;0.024486
1;0.995608;3.73775
1;0.151311;0.045017
1;0.126804;0.001238
1;0.983153;3.892763
1;0.772495;2.819376
1;0.784133;2.830665
1;0.056934;0.234633
1;0.425584;1.810782
1;0.998709;4.237235
1;0.707815;3.034768
1;0.413816;1.742106
1;0.217152;1.16925
1;0.360503;0.831165
1;0.977989;3.729376
1;0.507953;1.823205
1;0.920771;4.02197
1;0.210542;1.262939
1;0.928611;4.159518
1;0.580373;2.039114
1;0.84139;4.101837
1;0.68153;2.778672
1;0.292795;1.228284
1;0.456918;1.73662
1;0.134128;-0.195046
1;0.016241;-0.063215
1;0.691214;3.305268
1;0.582002;2.063627
1;0.303102;0.89884
1;0.622598;2.701692
1;0.525024;1.992909
1;0.996775;3.811393
1;0.881025;4.353857
1;0.723457;2.635641
1;0.676346;2.856311
1;0.254625;1.352682
1;0.488632;2.336459
1;0.519875;2.111651
1;0.160176;0.121726
1;0.609483;3.264605
1;0.531881;2.103446
1;0.321632;0.896855
1;0.845148;4.22085
1;0.012003;-0.217283
1;0.018883;-0.300577
1;0.071476;0.006014";

    public const string GlassData =
@"F1;F2;F3;F4;F5;F6;F7;F8;F10;Target
1.52101;13.64;4.49;1.1;71.78;0.06;8.75;0;0;1
1.51761;13.89;3.6;1.36;72.73;0.48;7.83;0;0;1
1.51618;13.53;3.55;1.54;72.99;0.39;7.78;0;0;1
1.51766;13.21;3.69;1.29;72.61;0.57;8.22;0;0;1
1.51742;13.27;3.62;1.24;73.08;0.55;8.07;0;0;1
1.51596;12.79;3.61;1.62;72.97;0.64;8.07;0;0.26;1
1.51743;13.3;3.6;1.14;73.09;0.58;8.17;0;0;1
1.51756;13.15;3.61;1.05;73.24;0.57;8.24;0;0;1
1.51918;14.04;3.58;1.37;72.08;0.56;8.3;0;0;1
1.51755;13;3.6;1.36;72.99;0.57;8.4;0;0.11;1
1.51571;12.72;3.46;1.56;73.2;0.67;8.09;0;0.24;1
1.51763;12.8;3.66;1.27;73.01;0.6;8.56;0;0;1
1.51589;12.88;3.43;1.4;73.28;0.69;8.05;0;0.24;1
1.51748;12.86;3.56;1.27;73.21;0.54;8.38;0;0.17;1
1.51763;12.61;3.59;1.31;73.29;0.58;8.5;0;0;1
1.51761;12.81;3.54;1.23;73.24;0.58;8.39;0;0;1
1.51784;12.68;3.67;1.16;73.11;0.61;8.7;0;0;1
1.52196;14.36;3.85;0.89;71.36;0.15;9.15;0;0;1
1.51911;13.9;3.73;1.18;72.12;0.06;8.89;0;0;1
1.51735;13.02;3.54;1.69;72.73;0.54;8.44;0;0.07;1
1.5175;12.82;3.55;1.49;72.75;0.54;8.52;0;0.19;1
1.51966;14.77;3.75;0.29;72.02;0.03;9;0;0;1
1.51736;12.78;3.62;1.29;72.79;0.59;8.7;0;0;1
1.51751;12.81;3.57;1.35;73.02;0.62;8.59;0;0;1
1.5172;13.38;3.5;1.15;72.85;0.5;8.43;0;0;1
1.51764;12.98;3.54;1.21;73;0.65;8.53;0;0;1
1.51793;13.21;3.48;1.41;72.64;0.59;8.43;0;0;1
1.51721;12.87;3.48;1.33;73.04;0.56;8.43;0;0;1
1.51768;12.56;3.52;1.43;73.15;0.57;8.54;0;0;1
1.51784;13.08;3.49;1.28;72.86;0.6;8.49;0;0;1
1.51768;12.65;3.56;1.3;73.08;0.61;8.69;0;0.14;1
1.51747;12.84;3.5;1.14;73.27;0.56;8.55;0;0;1
1.51775;12.85;3.48;1.23;72.97;0.61;8.56;0.09;0.22;1
1.51753;12.57;3.47;1.38;73.39;0.6;8.55;0;0.06;1
1.51783;12.69;3.54;1.34;72.95;0.57;8.75;0;0;1
1.51567;13.29;3.45;1.21;72.74;0.56;8.57;0;0;1
1.51909;13.89;3.53;1.32;71.81;0.51;8.78;0.11;0;1
1.51797;12.74;3.48;1.35;72.96;0.64;8.68;0;0;1
1.52213;14.21;3.82;0.47;71.77;0.11;9.57;0;0;1
1.52213;14.21;3.82;0.47;71.77;0.11;9.57;0;0;1
1.51793;12.79;3.5;1.12;73.03;0.64;8.77;0;0;1
1.51755;12.71;3.42;1.2;73.2;0.59;8.64;0;0;1
1.51779;13.21;3.39;1.33;72.76;0.59;8.59;0;0;1
1.5221;13.73;3.84;0.72;71.76;0.17;9.74;0;0;1
1.51786;12.73;3.43;1.19;72.95;0.62;8.76;0;0.3;1
1.519;13.49;3.48;1.35;71.95;0.55;9;0;0;1
1.51869;13.19;3.37;1.18;72.72;0.57;8.83;0;0.16;1
1.52667;13.99;3.7;0.71;71.57;0.02;9.82;0;0.1;1
1.52223;13.21;3.77;0.79;71.99;0.13;10.02;0;0;1
1.51898;13.58;3.35;1.23;72.08;0.59;8.91;0;0;1
1.5232;13.72;3.72;0.51;71.75;0.09;10.06;0;0.16;1
1.51926;13.2;3.33;1.28;72.36;0.6;9.14;0;0.11;1
1.51808;13.43;2.87;1.19;72.84;0.55;9.03;0;0;1
1.51837;13.14;2.84;1.28;72.85;0.55;9.07;0;0;1
1.51778;13.21;2.81;1.29;72.98;0.51;9.02;0;0.09;1
1.51769;12.45;2.71;1.29;73.7;0.56;9.06;0;0.24;1
1.51215;12.99;3.47;1.12;72.98;0.62;8.35;0;0.31;1
1.51824;12.87;3.48;1.29;72.95;0.6;8.43;0;0;1
1.51754;13.48;3.74;1.17;72.99;0.59;8.03;0;0;1
1.51754;13.39;3.66;1.19;72.79;0.57;8.27;0;0.11;1
1.51905;13.6;3.62;1.11;72.64;0.14;8.76;0;0;1
1.51977;13.81;3.58;1.32;71.72;0.12;8.67;0.69;0;1
1.52172;13.51;3.86;0.88;71.79;0.23;9.54;0;0.11;1
1.52227;14.17;3.81;0.78;71.35;0;9.69;0;0;1
1.52172;13.48;3.74;0.9;72.01;0.18;9.61;0;0.07;1
1.52099;13.69;3.59;1.12;71.96;0.09;9.4;0;0;1
1.52152;13.05;3.65;0.87;72.22;0.19;9.85;0;0.17;1
1.52152;13.05;3.65;0.87;72.32;0.19;9.85;0;0.17;1
1.52152;13.12;3.58;0.9;72.2;0.23;9.82;0;0.16;1
1.523;13.31;3.58;0.82;71.99;0.12;10.17;0;0.03;1
1.51574;14.86;3.67;1.74;71.87;0.16;7.36;0;0.12;2
1.51848;13.64;3.87;1.27;71.96;0.54;8.32;0;0.32;2
1.51593;13.09;3.59;1.52;73.1;0.67;7.83;0;0;2
1.51631;13.34;3.57;1.57;72.87;0.61;7.89;0;0;2
1.51596;13.02;3.56;1.54;73.11;0.72;7.9;0;0;2
1.5159;13.02;3.58;1.51;73.12;0.69;7.96;0;0;2
1.51645;13.44;3.61;1.54;72.39;0.66;8.03;0;0;2
1.51627;13;3.58;1.54;72.83;0.61;8.04;0;0;2
1.51613;13.92;3.52;1.25;72.88;0.37;7.94;0;0.14;2
1.5159;12.82;3.52;1.9;72.86;0.69;7.97;0;0;2
1.51592;12.86;3.52;2.12;72.66;0.69;7.97;0;0;2
1.51593;13.25;3.45;1.43;73.17;0.61;7.86;0;0;2
1.51646;13.41;3.55;1.25;72.81;0.68;8.1;0;0;2
1.51594;13.09;3.52;1.55;72.87;0.68;8.05;0;0.09;2
1.51409;14.25;3.09;2.08;72.28;1.1;7.08;0;0;2
1.51625;13.36;3.58;1.49;72.72;0.45;8.21;0;0;2
1.51569;13.24;3.49;1.47;73.25;0.38;8.03;0;0;2
1.51645;13.4;3.49;1.52;72.65;0.67;8.08;0;0.1;2
1.51618;13.01;3.5;1.48;72.89;0.6;8.12;0;0;2
1.5164;12.55;3.48;1.87;73.23;0.63;8.08;0;0.09;2
1.51841;12.93;3.74;1.11;72.28;0.64;8.96;0;0.22;2
1.51605;12.9;3.44;1.45;73.06;0.44;8.27;0;0;2
1.51588;13.12;3.41;1.58;73.26;0.07;8.39;0;0.19;2
1.5159;13.24;3.34;1.47;73.1;0.39;8.22;0;0;2
1.51629;12.71;3.33;1.49;73.28;0.67;8.24;0;0;2
1.5186;13.36;3.43;1.43;72.26;0.51;8.6;0;0;2
1.51841;13.02;3.62;1.06;72.34;0.64;9.13;0;0.15;2
1.51743;12.2;3.25;1.16;73.55;0.62;8.9;0;0.24;2
1.51689;12.67;2.88;1.71;73.21;0.73;8.54;0;0;2
1.51811;12.96;2.96;1.43;72.92;0.6;8.79;0.14;0;2
1.51655;12.75;2.85;1.44;73.27;0.57;8.79;0.11;0.22;2
1.5173;12.35;2.72;1.63;72.87;0.7;9.23;0;0;2
1.5182;12.62;2.76;0.83;73.81;0.35;9.42;0;0.2;2
1.52725;13.8;3.15;0.66;70.57;0.08;11.64;0;0;2
1.5241;13.83;2.9;1.17;71.15;0.08;10.79;0;0;2
1.52475;11.45;0;1.88;72.19;0.81;13.24;0;0.34;2
1.53125;10.73;0;2.1;69.81;0.58;13.3;3.15;0.28;2
1.53393;12.3;0;1;70.16;0.12;16.19;0;0.24;2
1.52222;14.43;0;1;72.67;0.1;11.52;0;0.08;2
1.51818;13.72;0;0.56;74.45;0;10.99;0;0;2
1.52664;11.23;0;0.77;73.21;0;14.68;0;0;2
1.52739;11.02;0;0.75;73.08;0;14.96;0;0;2
1.52777;12.64;0;0.67;72.02;0.06;14.4;0;0;2
1.51892;13.46;3.83;1.26;72.55;0.57;8.21;0;0.14;2
1.51847;13.1;3.97;1.19;72.44;0.6;8.43;0;0;2
1.51846;13.41;3.89;1.33;72.38;0.51;8.28;0;0;2
1.51829;13.24;3.9;1.41;72.33;0.55;8.31;0;0.1;2
1.51708;13.72;3.68;1.81;72.06;0.64;7.88;0;0;2
1.51673;13.3;3.64;1.53;72.53;0.65;8.03;0;0.29;2
1.51652;13.56;3.57;1.47;72.45;0.64;7.96;0;0;2
1.51844;13.25;3.76;1.32;72.4;0.58;8.42;0;0;2
1.51663;12.93;3.54;1.62;72.96;0.64;8.03;0;0.21;2
1.51687;13.23;3.54;1.48;72.84;0.56;8.1;0;0;2
1.51707;13.48;3.48;1.71;72.52;0.62;7.99;0;0;2
1.52177;13.2;3.68;1.15;72.75;0.54;8.52;0;0;2
1.51872;12.93;3.66;1.56;72.51;0.58;8.55;0;0.12;2
1.51667;12.94;3.61;1.26;72.75;0.56;8.6;0;0;2
1.52081;13.78;2.28;1.43;71.99;0.49;9.85;0;0.17;2
1.52068;13.55;2.09;1.67;72.18;0.53;9.57;0.27;0.17;2
1.5202;13.98;1.35;1.63;71.76;0.39;10.56;0;0.18;2
1.52177;13.75;1.01;1.36;72.19;0.33;11.14;0;0;2
1.52614;13.7;0;1.36;71.24;0.19;13.44;0;0.1;2
1.51813;13.43;3.98;1.18;72.49;0.58;8.15;0;0;2
1.518;13.71;3.93;1.54;71.81;0.54;8.21;0;0.15;2
1.51811;13.33;3.85;1.25;72.78;0.52;8.12;0;0;2
1.51789;13.19;3.9;1.3;72.33;0.55;8.44;0;0.28;2
1.51806;13;3.8;1.08;73.07;0.56;8.38;0;0.12;2
1.51711;12.89;3.62;1.57;72.96;0.61;8.11;0;0;2
1.51674;12.79;3.52;1.54;73.36;0.66;7.9;0;0;2
1.51674;12.87;3.56;1.64;73.14;0.65;7.99;0;0;2
1.5169;13.33;3.54;1.61;72.54;0.68;8.11;0;0;2
1.51851;13.2;3.63;1.07;72.83;0.57;8.41;0.09;0.17;2
1.51662;12.85;3.51;1.44;73.01;0.68;8.23;0.06;0.25;2
1.51709;13;3.47;1.79;72.72;0.66;8.18;0;0;2
1.5166;12.99;3.18;1.23;72.97;0.58;8.81;0;0.24;2
1.51839;12.85;3.67;1.24;72.57;0.62;8.68;0;0.35;2
1.51769;13.65;3.66;1.11;72.77;0.11;8.6;0;0;3
1.5161;13.33;3.53;1.34;72.67;0.56;8.33;0;0;3
1.5167;13.24;3.57;1.38;72.7;0.56;8.44;0;0.1;3
1.51643;12.16;3.52;1.35;72.89;0.57;8.53;0;0;3
1.51665;13.14;3.45;1.76;72.48;0.6;8.38;0;0.17;3
1.52127;14.32;3.9;0.83;71.5;0;9.49;0;0;3
1.51779;13.64;3.65;0.65;73;0.06;8.93;0;0;3
1.5161;13.42;3.4;1.22;72.69;0.59;8.32;0;0;3
1.51694;12.86;3.58;1.31;72.61;0.61;8.79;0;0;3
1.51646;13.04;3.4;1.26;73.01;0.52;8.58;0;0;3
1.51655;13.41;3.39;1.28;72.64;0.52;8.65;0;0;3
1.52121;14.03;3.76;0.58;71.79;0.11;9.65;0;0;3
1.51776;13.53;3.41;1.52;72.04;0.58;8.79;0;0;3
1.51796;13.5;3.36;1.63;71.94;0.57;8.81;0;0.09;3
1.51832;13.33;3.34;1.54;72.14;0.56;8.99;0;0;3
1.51934;13.64;3.54;0.75;72.65;0.16;8.89;0.15;0.24;3
1.52211;14.19;3.78;0.91;71.36;0.23;9.14;0;0.37;3
1.51514;14.01;2.68;3.5;69.89;1.68;5.87;2.2;0;5
1.51915;12.73;1.85;1.86;72.69;0.6;10.09;0;0;5
1.52171;11.56;1.88;1.56;72.86;0.47;11.41;0;0;5
1.52151;11.03;1.71;1.56;73.44;0.58;11.62;0;0;5
1.51969;12.64;0;1.65;73.75;0.38;11.53;0;0;5
1.51666;12.86;0;1.83;73.88;0.97;10.17;0;0;5
1.51994;13.27;0;1.76;73.03;0.47;11.32;0;0;5
1.52369;13.44;0;1.58;72.22;0.32;12.24;0;0;5
1.51316;13.02;0;3.04;70.48;6.21;6.96;0;0;5
1.51321;13;0;3.02;70.7;6.21;6.93;0;0;5
1.52043;13.38;0;1.4;72.25;0.33;12.5;0;0;5
1.52058;12.85;1.61;2.17;72.18;0.76;9.7;0.24;0.51;5
1.52119;12.97;0.33;1.51;73.39;0.13;11.27;0;0.28;5
1.51905;14;2.39;1.56;72.37;0;9.57;0;0;6
1.51937;13.79;2.41;1.19;72.76;0;9.77;0;0;6
1.51829;14.46;2.24;1.62;72.38;0;9.26;0;0;6
1.51852;14.09;2.19;1.66;72.67;0;9.32;0;0;6
1.51299;14.4;1.74;1.54;74.55;0;7.59;0;0;6
1.51888;14.99;0.78;1.74;72.5;0;9.95;0;0;6
1.51916;14.15;0;2.09;72.74;0;10.88;0;0;6
1.51969;14.56;0;0.56;73.48;0;11.22;0;0;6
1.51115;17.38;0;0.34;75.41;0;6.65;0;0;6
1.51131;13.69;3.2;1.81;72.81;1.76;5.43;1.19;0;7
1.51838;14.32;3.26;2.22;71.25;1.46;5.79;1.63;0;7
1.52315;13.44;3.34;1.23;72.38;0.6;8.83;0;0;7
1.52247;14.86;2.2;2.06;70.26;0.76;9.76;0;0;7
1.52365;15.79;1.83;1.31;70.43;0.31;8.61;1.68;0;7
1.51613;13.88;1.78;1.79;73.1;0;8.67;0.76;0;7
1.51602;14.85;0;2.38;73.28;0;8.76;0.64;0.09;7
1.51623;14.2;0;2.79;73.46;0.04;9.04;0.4;0.09;7
1.51719;14.75;0;2;73.02;0;8.53;1.59;0.08;7
1.51683;14.56;0;1.98;73.29;0;8.52;1.57;0.07;7
1.51545;14.14;0;2.68;73.39;0.08;9.07;0.61;0.05;7
1.51556;13.87;0;2.54;73.23;0.14;9.41;0.81;0.01;7
1.51727;14.7;0;2.34;73.28;0;8.95;0.66;0;7
1.51531;14.38;0;2.66;73.1;0.04;9.08;0.64;0;7
1.51609;15.01;0;2.51;73.05;0.05;8.83;0.53;0;7
1.51508;15.15;0;2.25;73.5;0;8.34;0.63;0;7
1.51653;11.95;0;1.19;75.18;2.7;8.93;0;0;7
1.51514;14.85;0;2.42;73.72;0;8.39;0.56;0;7
1.51658;14.8;0;1.99;73.11;0;8.28;1.71;0;7
1.51617;14.95;0;2.27;73.3;0;8.71;0.67;0;7
1.51732;14.95;0;1.8;72.99;0;8.61;1.55;0;7
1.51645;14.94;0;1.87;73.11;0;8.67;1.38;0;7
1.51831;14.39;0;1.82;72.86;1.41;6.47;2.88;0;7
1.5164;14.37;0;2.74;72.85;0;9.45;0.54;0;7
1.51623;14.14;0;2.88;72.61;0.08;9.18;1.06;0;7
1.51685;14.92;0;1.99;73.06;0;8.4;1.59;0;7
1.52065;14.36;0;2.02;73.42;0;8.44;1.64;0;7
1.51651;14.38;0;1.94;73.61;0;8.48;1.57;0;7
1.51711;14.23;0;2.08;73.36;0;8.62;1.67;0;7";

}