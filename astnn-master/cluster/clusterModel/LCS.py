# 将第一个字符串与最短的字符串交换
def swap(pStr, i):


    temp = pStr;
    pStr = pStr + i

def main(N,list):
    cin = N
    return


string * pStr;
pStr = new
string[N];
int
i, min;
int
maxLen = 1000;
# 找出输入的字符串中长度最小的串，并把最小串序号记在min中

for (i = 0; i < N; ++i){
                           cin >> * (pStr + i);
int len = ( * (pStr +i)).length(); // * 操作符与调用函数的.操作符优先级问题，.优先级高于 * ，所以必须加上()
if (len < maxLen){
maxLen = len;
min = i;
}
}
swap(pStr, min);
/ *
for (i = 0; i < N; ++i)
cout << * (pStr + i) << endl;
* /

int len0 = pStr[0].length();
int j, k, maxlen= 0;
string maxStr;
string tmpStr;
for (i = 0; i < len0 & & maxlen <= len0 - i -1; ++i)
{
for (j = 0; j < len0 & & maxlen <= len0 - i -j - 1; ++j)
{
tmpStr = pStr[0].substr(i, len0 - j); // 对字符串数组中第一个子串，求出其可能的子串值，如果剩余子串长度小于maxlen则不用去求了，for循环中给出了限制
// 将子串tmpStr与参与匹配的字符串比较，判断tmpStr是否为剩余串的子串，如果不是则break出循环
for (k = 1; k < N; ++k)
{
string::
    size_type
pos1 = pStr[k].find(tmpStr);
if (pos1 < pStr[k].length())
continue;
else
break;
}
if (k == N) // 说明子串tmpStr是其他参与匹配的子串的子串
{
if (tmpStr.length() > maxlen) // tmpStr如果是当前最大的子串，则记录下来
{
    maxlen = tmpStr.length();
maxStr = tmpStr;
}
}
}
}
cout << "最大公共子串为：";
cout << maxStr << endl;
delete[]
pStr;
return 0;
}
