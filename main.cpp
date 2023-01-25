#include<iostream>
#include <fstream>
#include<sstream>
#include <string>
#include<vector>
#include<cmath>
#include <algorithm>
#include<string>

#define PI 3.1415926535

using namespace std;
struct point {
	double x, y, z;
	friend ostream& operator<<(ostream& os, const point p) {
		os << "(" << p.x << "," << p.y << "," << p.z << ")";
		return os;
	}
};
void ReadData(vector<vector<double>>& vv, string a) {
	string file = "./" + a + ".txt";
	ifstream in(file);
	string line;
	while (getline(in, line)) {
		stringstream ss(line);
		string tmp;
		vector<double> v;
		while (getline(ss, tmp, ',')) {//按“，”隔开字符串
			v.push_back(stod(tmp));//stod: string->double
		}
		vv.push_back(v);
	}
	////打印数据集
	//for (auto row : vv) {
	//	for (auto col : row) {
	//		cout << col << "\t";
	//	}
	//	cout << endl;
	//}
	//cout << endl;
}
template<typename T>
void Print1(const vector<T> data) {
	int len = data.size();
	int line = 0;
	for (int i = 0; i < len; ++i)
	{
		line++;
		cout << data[i] << '\t';
		if (line == 10) { line = 0; printf("\n"); }
	}
	cout << endl;
}
template<typename T>
void Print2(const vector<vector<T>> data) {
	int width = data.size();
	const int len = data[0].size();
	double sum1 = 0, sum2 = 0;
	for (int i = 0; i < len; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			cout << data[j][i] << " ";
		}
		cout << endl;
	}
	cout << endl;
}
void Print_Res(vector<int> test_res) {
	int arr[4] = { 0 };
	for (int i = 0; i < test_res.size(); i++)
	{
		if (test_res[i] == 0) arr[0]++;
		if (test_res[i] == 1) arr[1]++;
		if (test_res[i] == 2) arr[2]++;
		if (test_res[i] == 3) arr[3]++;
	}
	printf("类别0：%d个  类别1：%d个  类别2：%d个  类别3：%d个", arr[0], arr[1], arr[2], arr[3]);
}
void cnt_trains(vector<vector<double>> train_data) {
	vector<int> origin = { 0,0,0,0 };
	for (int i = 0; i < 3000; ++i)
	{
		if (train_data[i][2] == 0) origin[0]++;
		if (train_data[i][2] == 1) origin[1]++;
		if (train_data[i][2] == 2) origin[2]++;
		if (train_data[i][2] == 3) origin[3]++;
	}
	cout << "训练集中各类的数目为：" << origin[0] << "  " << origin[1] << "  " << origin[2] << "  " << origin[3] << "  " << endl;
}

//计算先验概率
vector<vector<double>> Probability(vector<vector<double>> train_data) {
	vector<vector<double>> p_u;
	vector<double> p{ 0,0,0,0 };   //初始化p为3个0
	vector<double> u1{ 0,0,0,0 };  //第一列数据的均值，u1[0]表示类别‘0’中第一个特征的均值
	vector<double> u2{ 0,0,0,0 };
	vector<double> s1{ 0,0,0,0 };  //第一列数据的方差，s1[0]表示类别‘0’中第一个特征的方差
	vector<double> s2{ 0,0,0,0 };

	int len = train_data.size();
	for (int i = 0; i < len; ++i)
	{
		double a = train_data[i][0];
		double b = train_data[i][1];
		if (train_data[i][2] == 0)
		{
			p[0]++;
			u1[0] += a;
			u2[0] += b;
		}
		if (train_data[i][2] == 1)
		{
			p[1]++;
			u1[1] += a;
			u2[1] += b;
		}
		if (train_data[i][2] == 2)
		{
			p[2]++;
			u1[2] += a;
			u2[2] += b;
		}
		if (train_data[i][2] == 3)
		{
			p[3]++;
			u1[3] += a;
			u2[3] += b;
		}
	}
	for (int i = 0; i < 4; ++i)
	{
		u1[i] /= p[i];
		u2[i] /= p[i];
	}
	p_u.push_back(p);
	p_u.push_back(u1);
	p_u.push_back(u2);
	//计算sigma
	for (int k = 0; k < len; ++k)
	{
		double a = train_data[k][0];
		double b = train_data[k][1];
		if (train_data[k][2] == 0)
		{
			s1[0] += (a - u1[0]) * (a - u1[0]);
			s2[0] += (a - u2[0]) * (a - u2[0]);
		}
		if (train_data[k][2] == 1)
		{
			s1[1] += (a - u1[1]) * (a - u1[1]);
			s2[1] += (a - u2[1]) * (a - u2[1]);
		}
		if (train_data[k][2] == 2)
		{
			s1[2] += (a - u1[2]) * (a - u1[2]);
			s2[2] += (a - u2[2]) * (a - u2[2]);
		}
		if (train_data[k][2] == 3)
		{
			s1[3] += (a - u1[3]) * (a - u1[3]);
			s2[3] += (a - u2[3]) * (a - u2[3]);
		}
	}
	for (int k = 0; k < 4; ++k)
	{
		s1[k] = s1[k] / p[k];
		s2[k] = s2[k] / p[k];
	}
	for (int i = 0; i < 4; ++i) p[i] = (p[i] + 1) / (len + 4);

	p_u.push_back(s1);
	p_u.push_back(s2);
	return p_u;
}

//计算P(x_j|c_i)：类别i的情况下特征是x_j的概率
double P_Gauss(double x, double u, double s_f) {
	return exp(-(((x - u) * (x - u)) / (2 * s_f))) / sqrt(2 * PI * s_f);
}

//测试集
int Find_Max(double* pp, int len = 4) {
	int max_i = 0;
	double max = 0;
	for (int i = 0; i < len; ++i)
	{
		if (pp[i] > max)
		{
			max = pp[i];
			max_i = i;
		}
	}
	return max_i;
}
vector<int> Test(
	vector<vector<double>> test_data, vector<double> p_prior,
	vector<double> u1, vector<double> u2, vector<double> s1_f, vector<double> s2_f
) {
	vector<int> CLUSTER;
	for (int i = 0; i < test_data.size(); ++i)
	{
		vector<double> p0{ 0,0 };
		vector<double> p1{ 0,0 };
		vector<double> p2{ 0,0 };
		vector<double> p3{ 0,0 };
		double pp[4];
		p0[0] = P_Gauss(test_data[i][0], u1[0], s1_f[0]);
		p0[1] = P_Gauss(test_data[i][1], u2[0], s2_f[0]);
		pp[0] = p_prior[0] * p0[0] * p0[1];
		p1[0] = P_Gauss(test_data[i][0], u1[1], s1_f[1]);
		p1[1] = P_Gauss(test_data[i][1], u2[1], s2_f[1]);
		pp[1] = p_prior[1] * p1[0] * p1[1];
		p2[0] = P_Gauss(test_data[i][0], u1[2], s1_f[2]);
		p2[1] = P_Gauss(test_data[i][1], u2[2], s2_f[2]);
		pp[2] = p_prior[2] * p2[0] * p2[1];
		p3[0] = P_Gauss(test_data[i][0], u1[3], s1_f[3]);
		p3[1] = P_Gauss(test_data[i][1], u2[3], s2_f[3]);
		pp[3] = p_prior[3] * p3[0] * p3[1];
		int max_i = Find_Max(pp);
		CLUSTER.push_back(max_i);
	}
	return CLUSTER;
}

void BayesClassifier(vector<vector<double>> train_data, vector<vector<double>> test_data) {
	cout << "朴素贝叶斯法-------------------------------" << endl;
	//训练
	//计算先验概率
	vector<vector<double>> p_u = Probability(train_data);
	vector<double> p_prior = p_u.at(0);
	//极大似然法计算连续性属性P(x_{j}|c)
	vector<double> u1 = p_u.at(1);
	vector<double> u2 = p_u.at(2);
	vector<double> sigma1_fang = p_u.at(3);
	vector<double> sigma2_fang = p_u.at(4);
	cout << "特征1均值" << endl;
	Print1(u1);
	cout << "特征1方差" << endl;
	Print1(sigma1_fang);
	//测试
	vector<int> test_res = Test(test_data, p_prior, u1, u2, sigma1_fang, sigma2_fang);
	cout << "分类结果为：" << endl;
	printf("分类情况明细：\n");
	Print1(test_res);  
	Print_Res(test_res);
	printf("\n\n\n");
}
///////////////////////////////////////////////////////////////////////////////////////////////////
class Node {
public:
	int id;
	double value;
	Node* leftBranch;
	Node* rightBranch;
	Node()
	{
		this->id = -1;
		this->value = -1;
		this->leftBranch = NULL;
		this->rightBranch = NULL;
	}
};

struct Order2Sort {
	double score1;
	double score2;
	int cluster;
};

void Vec2Stru(Order2Sort* s, vector<vector<double>> train_data) {
	for (int i = 0; i < train_data.size(); ++i)
	{
		s[i].score1 = train_data[i][0];
		s[i].score2 = train_data[i][1];
		s[i].cluster = train_data[i][2];
	}
}
//排序后将结构体转变会矩阵以提高计算效率
vector<vector<double>> Stru2Vec(Order2Sort* s, int len) {
	vector<double> a = { 0,0,0 };
	vector<vector<double>> b;
	for (int i = 0; i < len; ++i)
	{
		a[0] = s[i].score1;
		a[1] = s[i].score2;
		a[2] = s[i].cluster;
		b.push_back(a);
	}
	return b;
}
bool cmp1(Order2Sort s1, Order2Sort s2) {
	return s1.score1 < s2.score1;
}
bool cmp2(Order2Sort s1, Order2Sort s2) {
	return s1.score2 < s2.score2;
}
vector<vector<double>> get_Means(vector<vector<double>> dataSet_feat1, vector<vector<double>> dataSet_feat2) {
	vector<double> feat1;
	vector<double> feat2;
	for (int i = 0; i < dataSet_feat1.size(); ++i)
	{
		feat1.push_back(dataSet_feat1[i][0]);
		feat2.push_back(dataSet_feat1[i][1]);
	}
	vector<vector<double>> f;
	vector<double> f1, f2;
	for (int i = 0; i < feat1.size() - 1; ++i)
	{
		double tf1 = (feat1[i] + feat1[i + 1]) / 2;
		double tf2 = (feat2[i] + feat2[i + 1]) / 2;
		f1.push_back(tf1);
		f2.push_back(tf2);
	}
	f.push_back(f1);
	f.push_back(f2);
	return f;
}
//对特征进行升序排序并计算（a1+a2）/2
vector<vector<double>> get_MidnumVec(vector<vector<double>> dataset) {
	const int len = dataset.size();
	Order2Sort* s = new Order2Sort[len];
	Vec2Stru(s, dataset);
	sort(s, s + len, cmp1);//对特征1进行升序排列
	vector<vector<double>> dataSet_feat1 = Stru2Vec(s, len);//特征1升序
	sort(s, s + len, cmp2);
	vector<vector<double>> dataSet_feat2 = Stru2Vec(s, len);//特征2升序
	vector<vector<double>> mid_number = get_Means(dataSet_feat1, dataSet_feat2);
	delete[] s;
	return mid_number;
}

//计算信息熵
vector<double> calPro(vector<vector<double>> dataSet) {
	vector<double> p = { 0,0,0,0 };
	int len = dataSet.size();
	int sum = 0;
	for (int i = 0; i < len; ++i)
	{
		if (dataSet[i][2] == 0) p[0]++;
		if (dataSet[i][2] == 1) p[1]++;
		if (dataSet[i][2] == 2) p[2]++;
		if (dataSet[i][2] == 3) p[3]++;
		sum++;
	}
	p[0] /= sum; p[1] /= sum; p[2] /= sum; p[3] /= sum;
	return p;
}
double calcShannonEnt(vector<vector<double>> dataSet) {
	vector<double> p = calPro(dataSet);
	int len = p.size();
	int i = 0;
	double sum = 0;
	const int a = 2; //底数为2
	while (i < len)
	{
		if (p[i] != 0)
			sum += (p[i] * (log(p[i]) / log(a)));
		i++;
	}
	return -sum;
}
double calcShannonEnt(vector<double> p) {
	int len = p.size();
	int i = 0;
	double sum = 0;
	const int a = 2; //底数为2
	while (i < len)
	{
		if (p[i] != 0)
			sum += (p[i] * (log(p[i]) / log(a)));
		i++;
	}
	return -sum;
}

//id的取值为0或1，表示是第一个属性
struct Feat_Id
{
	double feature = -1;
	int id = -1;
};
Feat_Id chooseBestFeatureToSplit(vector<vector<double>> dataSet, vector<vector<double>> mid_number) {
	Feat_Id fead_id;
	double baseEntropy = calcShannonEnt(dataSet);
	double bestInfoGain = 0.0;
	double bestFeature = -1;
	int id = -1; //记录属性号（其取值为0或1）
	//当attribute为0时处理第一列特征，为1时处理第二列特征
	for (int attribute = 0; attribute < 2; attribute++)
	{
		int j = 0;
		while (j < mid_number.at(attribute).size())
		{
			vector<double> p1 = { 0,0,0,0 }; //左边
			vector<double> p2 = { 0,0,0,0 }; //右边
			double spilit = mid_number.at(attribute)[j];
			for (int i = 0; i < dataSet.size(); ++i)
			{
				if (dataSet[i][attribute] <= spilit)  //通过spilit来进行划分，小于这个值分到左边，大于分到右边
				{
					if (dataSet[i][2] == 0) p1[0]++;
					if (dataSet[i][2] == 1) p1[1]++;
					if (dataSet[i][2] == 2) p1[2]++;
					if (dataSet[i][2] == 3) p1[3]++;
				}
				else
				{
					if (dataSet[i][2] == 0) p2[0]++;
					if (dataSet[i][2] == 1) p2[1]++;
					if (dataSet[i][2] == 2) p2[2]++;
					if (dataSet[i][2] == 3) p2[3]++;
				}
			}
			double sum1 = p1[0] + p1[1] + p1[2] + p1[3];
			double sum2 = p2[0] + p2[1] + p2[2] + p2[3];
			double sum = sum1 + sum2;
			p1[0] /= sum1; p1[1] /= sum1; p1[2] /= sum1; p1[3] /= sum1;
			p2[0] /= sum2; p2[1] /= sum2; p2[2] /= sum2; p2[3] /= sum2;
			double newEntropy = (sum1 / sum) * calcShannonEnt(p1) + (sum2 / sum) * calcShannonEnt(p2);
			double infoGain = baseEntropy - newEntropy;
			//cout << "infoGain=" << infoGain << endl;
			//cout << "calcShannonEnt(p1)=" << calcShannonEnt(p1) << endl;
			//cout << "baseEntropy=" << baseEntropy << endl;
			//cout << "newEntropy=" << newEntropy << endl;
			if (infoGain > bestInfoGain)
			{
				bestInfoGain = infoGain;
				bestFeature = spilit;
				id = attribute;
			}
			j++;
		}
	}
	fead_id.feature = bestFeature;
	fead_id.id = id;
	return fead_id;
}
//分割后第0列为小数据，第1列为大数据
vector<vector<vector<double>>> splitDataSet(vector<vector<double>> dataSet, int id, double val) {
	vector<vector<vector<double>>> sp;
	vector<vector<double>> dataSet_feat;
	vector<vector<double>> left;
	vector<vector<double>> right;
	const int len = dataSet.size();
	Order2Sort* s = new Order2Sort[len];
	Vec2Stru(s, dataSet);
	if (id == 0)
	{
		sort(s, s + len, cmp1);//对特征1进行升序排列
		dataSet_feat = Stru2Vec(s, len);//特征1升序
		delete[] s;
	}
	else if (id == 1)
	{
		sort(s, s + len, cmp2);//对特征2进行升序排列
		dataSet_feat = Stru2Vec(s, len);//特征2升序
		delete[] s;
	}
	else
	{
		cout << "ERROR IN splitDataSet \"id=-1\"";
		exit(1);
	}
	for (int i = 0; i < len; ++i)
	{
		if (dataSet_feat[i][id] <= val)
		{
			left.push_back(dataSet_feat[i]);
		}
		if (dataSet_feat[i][id] > val)
		{
			right.push_back(dataSet_feat[i]);
		}
	}
	sp.push_back(left);
	sp.push_back(right);
	return sp;
}
vector<Node*> MyTrees;

void deleteTrees() {
	cout << "删除堆区缓存空间……" << endl;
	vector<Node*>().swap(MyTrees);
	cout << "trees.size = " << MyTrees.size() << endl;
	cout << "trees.capacity = " << MyTrees.capacity() << endl;
	if (!MyTrees.capacity())  cout << "成功释放空间!!!" << endl;
}
Node* createTree(Node* tree, vector<vector<double>> dataSet) {
	const int len = dataSet.size();
	vector<double> labels = { 0,0,0,0 };
	for (int i = 0; i < len; ++i)
	{
		if (dataSet[i][2] == 0) labels[0]++;
		if (dataSet[i][2] == 1) labels[1]++;
		if (dataSet[i][2] == 2) labels[2]++;
		if (dataSet[i][2] == 3) labels[3]++;
	}
	for (int i = 0; i < 4; ++i)
	{
		if (labels[i] == len)
		{
			tree->value = i;
			return tree;
		}
	}
	vector<vector<double>> mid_numVec = get_MidnumVec(dataSet);
	Feat_Id fd = chooseBestFeatureToSplit(dataSet, mid_numVec);
	tree->id = fd.id;
	tree->value = fd.feature;
	//cout << "feature=" << fd.feature << "   id=" << fd.id << endl;
	vector<vector<vector<double>>> sp = splitDataSet(dataSet, fd.id, fd.feature);
	vector<vector<double>> left = sp.at(0);
	vector<vector<double>> right = sp.at(1);
	MyTrees.push_back(new Node());
	tree->leftBranch = createTree(MyTrees.back(), left);
	MyTrees.push_back(new Node());
	tree->rightBranch = createTree(MyTrees.back(), right);
	return tree;
}
int classify(Node* inputtree, vector<double> one_testdata) {
	double firstattr = inputtree->value;
	int id = inputtree->id;
	//cout << "value=" << firstattr << "  id=" << id << endl;
	if (id == -1) return firstattr;
	if (one_testdata[id] <= firstattr)
	{
		return classify(inputtree->leftBranch, one_testdata);
	}
	if (one_testdata[id] > firstattr)
	{
		return classify(inputtree->rightBranch, one_testdata);
	}
}
string spaces = "";
void Print_Tree(Node tree) {
	printf("%d, %lf\n", tree.id, tree.value);
	spaces += "  ";
	if (tree.rightBranch != nullptr)
	{
		printf("%s", spaces.c_str());
		Print_Tree(*tree.rightBranch);
	}

	spaces.pop_back();
	spaces.pop_back();
	if (tree.leftBranch != nullptr)
	{
		printf("%s", spaces.c_str());
		Print_Tree(*tree.leftBranch);
	}
}
void decisionTree(vector<vector<double>> train_data, vector<vector<double>> test_data) {
	cout << "决策树法-------------------------------" << endl;
	Node Tree;  //构建决策树
	createTree(&Tree, train_data);
	vector<int> res;
	for (int i = 0; i < test_data.size(); ++i)
	{
		res.push_back(classify(&Tree, test_data[i]));
	}
	//Print_Tree(Tree); cout << endl;  //打印树结构
	printf("分类情况明细：\n");
	Print1(res); 
	Print_Res(res);
	deleteTrees();
	printf("\n\n\n");
}

///////////////////////////////////////////////////////////////////////////////////////////////////
vector<vector<vector<double>>> splitDataSet(vector<vector<double>> dataSet) {
	int len = dataSet.size();
	vector<vector<double>> data01;
	vector<vector<double>> data02;
	vector<vector<double>> data03;
	vector<vector<double>> data12;
	vector<vector<double>> data13;
	vector<vector<double>> data23;
	for (int i = 0; i < len; ++i)
	{
		if (dataSet[i][2] == 0)
		{
			data01.push_back(dataSet[i]);
			data02.push_back(dataSet[i]);
			data03.push_back(dataSet[i]);
		}
		if (dataSet[i][2] == 1)
		{
			data01.push_back(dataSet[i]);
			data12.push_back(dataSet[i]);
			data13.push_back(dataSet[i]);
		}
		if (dataSet[i][2] == 2)
		{
			data02.push_back(dataSet[i]);
			data12.push_back(dataSet[i]);
			data23.push_back(dataSet[i]);
		}
		if (dataSet[i][2] == 3)
		{
			data03.push_back(dataSet[i]);
			data13.push_back(dataSet[i]);
			data23.push_back(dataSet[i]);
		}
	}
	vector<vector<vector<double>>> data = { data01,data02,data03,data12,data13,data23 };
	return data;
}

double distance(vector<double> w, vector<double> x) {
	if (x.size() == 2) x.push_back(1); //该行用于测试集，因为测试集仅两列，补充此行代码可简化后续测试
	double wx = (w[0] * x[0] + w[1] * x[1] + w[2]) * x[2];
	return wx;
}
double yita = 1;
void SGD(vector<double>& w, vector<double>x) {
	double c = yita * x[2];
	w[0] += c * x[0];
	w[1] += c * x[1];
	w[2] += c;
}
int cnt = 0;
//数据传的是哪两类的，w就传入对应的类别。  如dataSet传data01，则w传w01，clus传0（表示类别0为正例）
bool updata(vector<vector<double>> dataSet, vector<double>& w, int clus) {
	printf("第%d轮迭代……\n", ++cnt);
	bool signal = true;
	int len = dataSet.size();
	for (int i = 0; i < len; ++i)
	{
		int cluster = dataSet[i][2];
		vector<double> tempX = dataSet[i];
		if (cluster == clus) tempX[2] = 1;
		else tempX[2] = -1;

		double w_x = distance(w, tempX);
		if (w_x > 0) continue;
		else
		{
			SGD(w, tempX);
			signal = false;
		}
	}
	printf("本轮权重：%lf\t%lf\t%lf\n", w[0], w[1], w[2]);
	return signal;
}
// 0  1  2  3  4  5
//01 02 03 12 13 23
int selectCluster(vector<vector<double>> ww, vector<double> testx) {
	if (distance(ww[0], testx) > 0 && distance(ww[1], testx) > 0 && distance(ww[2], testx) > 0)
		return 0;
	if (distance(ww[0], testx) <= 0 && distance(ww[3], testx) > 0 && distance(ww[4], testx) > 0)
		return 1;
	if (distance(ww[1], testx) <= 0 && distance(ww[3], testx) <= 0 && distance(ww[5], testx) > 0)
		return 2;
	if (distance(ww[2], testx) <= 0 && distance(ww[4], testx) <= 0 && distance(ww[5], testx) <= 0)
		return 3;
	else
	{
		cout << "selectCluster()函数中止" << endl;
		return -1;
	}
}

vector<int> Test(vector<vector<double>> testdata, vector<vector<double>> ww) {
	vector<int> res;
	for (int i = 0; i < testdata.size(); ++i)
	{
		res.push_back(selectCluster(ww, testdata[i]));
	}
	return res;
}
void perceptron(vector<vector<double>> train_data, vector<vector<double>> test_data) {
	vector<double> w01 = { 0,0,0 };
	vector<double> w02 = { 0,0,0 };
	vector<double> w03 = { 0,0,0 };
	vector<double> w12 = { 0,0,0 };
	vector<double> w13 = { 0,0,0 };
	vector<double> w23 = { 0,0,0 };
	vector<vector<double>> ww = { w01,w02,w03,w12,w13,w23 };
	vector<vector<vector<double>>> dataset = splitDataSet(train_data);
	cout << "############################ w01 ############################" << endl;
	cnt = 0;
	while (!updata(dataset[0], ww[0], 0) && cnt < 100);
	cout << "############################ w02 ############################" << endl;
	cnt = 0;
	while (!updata(dataset[1], ww[1], 0) && cnt < 100);
	cout << "############################ w03 ############################" << endl;
	cnt = 0;
	while (!updata(dataset[2], ww[2], 0) && cnt < 100);
	cout << "############################ w12 ############################" << endl;
	cnt = 0;
	while (!updata(dataset[3], ww[3], 1) && cnt < 100);
	cout << "############################ w13 ############################" << endl;
	cnt = 0;
	while (!updata(dataset[4], ww[4], 1) && cnt < 100);
	cout << "############################ w23 ############################" << endl;
	cnt = 0;
	while (!updata(dataset[5], ww[5], 2) && cnt < 100);
	vector<int> res = Test(test_data, ww);
	printf("分类情况明细：\n");
	Print1(res);
	Print_Res(res);
}
void main(int argc, char** argv)
{
	vector<vector<double>> train_data; //训练集数据
	vector<vector<double>> test_data;  //测试集数据
	ReadData(train_data, "train");
	ReadData(test_data, "test");
	//cout << "原始数据类别情况" << endl;
	//cnt_trains(train_data);
	//cout << "贝叶斯回代验证" << endl;
	//BayesClassifier(train_data, train_data);
	//cout << "决策树回代验证" << endl;
	//decisionTree(train_data, train_data);

	//朴素贝叶斯法
	//BayesClassifier(train_data, test_data);
	//决策树法
	//decisionTree(train_data, test_data);
	//感知机
	perceptron(train_data, test_data);


}