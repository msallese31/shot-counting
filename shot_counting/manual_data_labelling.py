import data_munging as dm
import numpy as np

def label_data():
	data_file_path = 'data/'
	pattern_file = data_file_path + '3shots2.txt'
	mag = dm.mag_from_file(pattern_file)
	p1 = mag[101:201]
	p2 = mag[450:550]
	p3 = mag[860:960]
	patterns = np.vstack((p1, p2, p3))

	np1 = mag[0:100]
	np2 = mag[210:310]
	np3 = mag[643:743]
	np4 = mag[1000:1100]
	np5 = mag[1140:1240]
	non_patterns = np.vstack((np1, np2, np3, np4, np5))

	pattern_file1 = data_file_path + '3shots3.txt'
	mag1 = dm.mag_from_file(pattern_file1)
	p4 = mag1[80:180]
	p5 = mag1[300:400]
	p6 = mag1[590:690]

	patterns = np.vstack((patterns, p4, p5, p6))
	
	np6 = mag1[190:290]
	np7 = mag1[420:520]
	np8 = mag1[700:800]
	np9 = mag1[800:900]

	non_patterns = np.vstack((non_patterns, np6, np7, np8, np9))

	pattern_file2 = data_file_path + '3shots4.txt'

	mag2 = dm.mag_from_file(pattern_file2)
	p7 = mag2[80:180]
	p8 = mag2[530:630]
	p9 = mag2[930:1030]

	patterns = np.vstack((patterns, p7, p8, p9))
	
	np10 = mag2[200:300]
	np11 = mag2[301:401]
	np12 = mag2[402:502]
	np13 = mag2[700:800]
	np14 = mag2[801:901]
	np15 = mag2[1100:1200]

	non_patterns = np.vstack((non_patterns, np10, np11, np12, np13, np14, np15))

	pattern_file3 = data_file_path + '3shots.txt'
	mag3 = dm.mag_from_file(pattern_file3)

	p10 = mag3[120:220]
	p11 = mag3[435:535]
	p12 = mag3[870:970]

	patterns = np.vstack((patterns, p10, p11, p12))

	np16 = mag3[1:101]
	np17 = mag3[200:300]
	np18 = mag3[301:401]
	np19 = mag3[750:850]

	non_patterns = np.vstack((non_patterns, np16, np17, np18, np19))

	pattern_file4 = data_file_path + '5shots2.txt'
	mag4 = dm.mag_from_file(pattern_file4)

	p13 = mag4[115:215]
	p14 = mag4[430:530]
	p15 = mag4[815:915]
	p16 = mag4[1330:1430]
	p17 = mag4[1530:1630]

	patterns = np.vstack((patterns, p13, p14, p15, p16, p17))

	np20 = mag4[245:345]
	np21 = mag4[580:680]
	np22 = mag4[940:1040]
	np23 = mag4[1041:1141]
	np24 = mag4[1152:1252]
	np25 = mag4[1700:1800]

	non_patterns = np.vstack((non_patterns, np20, np21, np22, np23, np24, np25))

	pattern_file5 = data_file_path + '5shots3.txt'
	mag5 = dm.mag_from_file(pattern_file5)
	mag5[mag5 > 90] = 90

	p18 = mag5[60:160]
	p19 = mag5[594:694]
	p20 = mag5[1070:1170]
	p21 = mag5[1312:1412]
	p22 = mag5[1580:1680]

	patterns = np.vstack((patterns, p18, p19, p20, p21, p22))

	np26 = mag5[200:300]
	np27 = mag5[301:401]
	np28 = mag5[730:830]
	np29 = mag5[900:1000]
	np30 = mag5[1740:1840]
	np31 = mag5[1900:2000]

	non_patterns = np.vstack((non_patterns, np26, np27, np28, np29, np30, np31))

	pattern_file6 = data_file_path + '5shots4.txt'
	mag6 = dm.mag_from_file(pattern_file6)

	p23 = mag6[66:166]
	p24 = mag6[370:470]
	p25 = mag6[540:640]
	p26 = mag6[2015:2115]
	p27 = mag6[2195:2295]

	patterns = np.vstack((patterns, p23, p24, p25, p26, p27))

	np32 = mag6[734:834]
	np33 = mag6[900:1000]
	np34 = mag6[1001:1101]
	np35 = mag6[1102:1202]
	np36 = mag6[1203:1303]
	np37 = mag6[1304:1404]
	np38 = mag6[1405:1505]
	np39 = mag6[1506:1606]
	np40 = mag6[1607:1707]
	np41 = mag6[1708:1808]
	np42 = mag6[1809:1909]

	non_patterns = np.vstack((non_patterns, np32, np33, np34, np35, np36, np37, np38, np39, np40, np41, np42))

	pattern_file7 = data_file_path + '5shots.txt'
	mag7 = dm.mag_from_file(pattern_file7)
	mag7[mag7 > 90] = 90

	p28 = mag7[202:302]
	p29 = mag7[585:685]
	p30 = mag7[803:903]
	p31 = mag7[1015:1115]
	p32 = mag7[1425:1525]

	patterns = np.vstack((patterns, p28, p29, p30, p31, p32))

	np43 = mag7[1:101]
	np44 = mag7[355:455]
	np45 = mag7[1151:1251]
	np46 = mag7[1252:1352]
	np47 = mag7[1550:1650]

	non_patterns = np.vstack((non_patterns, np43, np44, np45, np46, np47))

	X = np.vstack((patterns,non_patterns))
	Y = np.zeros(shape=(79,))

	Y[0:31] = 1

	return X, Y