﻿namespace SharpLearning.InputOutput.Test;

public static class DataSetUtilities
{
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

    public const string TimeData1 =
@"Date;Open;High;Low;Close;Volume;Adj Close
2014-04-29;38.01;39.68;36.80;38.00;294200;38.00
2014-04-28;38.26;39.36;37.30;37.83;361900;37.83
2014-04-25;38.33;39.04;37.88;38.00;342900;38.00
2014-04-24;39.33;39.59;37.91;38.82;362200;38.82
2014-04-23;38.98;39.58;38.50;38.88;245800;38.88
2014-04-22;38.43;39.79;38.31;38.99;358000;38.99
2014-04-21;38.05;38.74;37.77;38.41;316800;38.41
2014-04-17;37.25;38.24;36.92;38.05;233700;38.05
2014-04-16;36.37;37.27;36.17;37.26;144800;37.26
2014-04-15;36.08;36.74;35.09;36.05;223100;36.05
2014-04-14;36.55;36.90;35.33;36.02;296100;36.02
2014-04-11;36.26;37.09;36.08;36.13;282700;36.13
2014-04-10;37.06;37.16;36.13;36.46;309800;36.46
2014-04-09;36.08;37.26;35.66;37.13;209400;37.13
2014-04-08;35.50;36.16;35.28;35.85;215700;35.85
2014-04-07;36.49;37.30;35.27;35.48;312400;35.48
2014-04-04;38.39;38.90;36.60;36.93;306500;36.93
2014-04-03;38.62;39.78;37.90;38.14;269800;38.14
2014-04-02;38.66;38.84;38.04;38.56;398200;38.56
2014-04-01;37.21;38.65;36.58;38.49;410900;38.49";

    public const string TimeData2 =
@"Date;Open;High;Low;Close;Volume;Adj Close
2014-05-05;20.89;20.99;20.55;20.89;62200;20.89
2014-05-02;21.08;21.50;20.85;21.09;76600;21.09
2014-05-01;21.17;21.28;20.62;20.97;102600;20.97
2014-04-30;21.78;21.89;21.14;21.27;171800;21.27
2014-04-29;22.05;22.44;21.72;21.78;81900;21.78
2014-04-28;21.79;22.00;21.46;21.90;71100;21.90
2014-04-25;22.10;22.48;21.67;21.78;77500;21.78
2014-04-24;22.61;22.70;22.20;22.23;48700;22.23
2014-04-23;22.26;22.95;22.16;22.60;99400;22.60
2014-04-22;22.19;22.70;22.13;22.48;69200;22.48
2014-04-21;22.28;22.54;22.05;22.24;31100;22.24
2014-04-17;22.30;22.40;22.15;22.26;47400;22.26
2014-04-16;22.59;22.74;22.09;22.35;46600;22.35
2014-04-15;22.46;22.74;21.95;22.35;40800;22.35
2014-04-14;22.65;22.82;22.16;22.45;84600;22.45
2014-04-11;22.31;22.69;22.28;22.43;66600;22.43
2014-04-10;23.11;23.25;22.39;22.56;88800;22.56
2014-04-09;23.15;23.30;22.95;23.18;58600;23.18
2014-04-08;23.04;23.68;23.00;23.11;56200;23.11
2014-04-07;23.41;23.73;23.01;23.09;61500;23.09
2014-04-04;24.00;24.05;23.37;23.44;188500;23.44
2014-04-03;23.97;23.97;23.77;23.90;43600;23.90
2014-04-02;23.70;23.92;23.51;23.88;74700;23.88
2014-04-01;23.34;23.87;23.13;23.75;146100;23.75";

    public const string TimeData21 =
@"Date;OpenOther;High;Low;CloseOther;Volume;Adj Close
2014-05-05;20.89;20.99;20.55;20.89;62200;20.89
2014-05-02;21.08;21.50;20.85;21.09;76600;21.09
2014-05-01;21.17;21.28;20.62;20.97;102600;20.97
2014-04-30;21.78;21.89;21.14;21.27;171800;21.27
2014-04-29;22.05;22.44;21.72;21.78;81900;21.78
2014-04-28;21.79;22.00;21.46;21.90;71100;21.90
2014-04-25;22.10;22.48;21.67;21.78;77500;21.78
2014-04-24;22.61;22.70;22.20;22.23;48700;22.23
2014-04-23;22.26;22.95;22.16;22.60;99400;22.60
2014-04-22;22.19;22.70;22.13;22.48;69200;22.48
2014-04-21;22.28;22.54;22.05;22.24;31100;22.24
2014-04-17;22.30;22.40;22.15;22.26;47400;22.26
2014-04-16;22.59;22.74;22.09;22.35;46600;22.35
2014-04-15;22.46;22.74;21.95;22.35;40800;22.35
2014-04-14;22.65;22.82;22.16;22.45;84600;22.45
2014-04-11;22.31;22.69;22.28;22.43;66600;22.43
2014-04-10;23.11;23.25;22.39;22.56;88800;22.56
2014-04-09;23.15;23.30;22.95;23.18;58600;23.18
2014-04-08;23.04;23.68;23.00;23.11;56200;23.11
2014-04-07;23.41;23.73;23.01;23.09;61500;23.09
2014-04-04;24.00;24.05;23.37;23.44;188500;23.44
2014-04-03;23.97;23.97;23.77;23.90;43600;23.90
2014-04-02;23.70;23.92;23.51;23.88;74700;23.88
2014-04-01;23.34;23.87;23.13;23.75;146100;23.75";

}