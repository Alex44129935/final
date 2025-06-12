# 載入必要套件
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#from haohaninfo.MicroTest import microtest_db
import numpy as np
import time
#import haohaninfo
#GOC = haohaninfo.GOrder.GOCommand()
#GOQ = haohaninfo.GOrder.GOQuote()

# 下單部位管理物件
class Record():
    
    # #### 交易成本(單次)
    # ### 價差(流動性風險):(買賣價差/2)/一口小台的金額.  買賣價差(Bid/Ask Spread)
    # G_spread = 3.6280082234853065666948845084049e-4
    # ### 交易稅: 期貨(進出場皆收取): 契約價值*0.00002; 股票(出場收取): 股票價值*0.003
    # G_tax_f = 0.00002    ## 期貨進出場各收取交易稅 0.00002
    # G_tax_s = 0.003    ## 股票在出場時收取0.003
    # ### 手續費:手續費/一口小台的金額
    # G_commission_f = 4.4109538687741224039698584818967e-5  ## 期貨手續費是買賣各收一次. 期貨手續費是固定金額.
    # G_commission_s = 0.001425  ## 股票手續費是買賣各收一次0.1425%.  股票手續費是算%數.
    # ### 完整交易成本(當價值為 1 時)
    # G_tradecost_f = 0*G_spread + 2*G_tax_f + 2*G_commission_f  ## 期貨
    # G_tradecost_s = 0*G_spread + G_tax_s + 2*G_commission_s    ## 股票

    
    def __init__(self, G_spread=3.628e-4, G_tax=0.00002, G_commission=4.411e-5, isFuture=True):   ## 建構子
        ## 期貨還是股票
        self.isFuture = isFuture
        ## 儲存績效
        self.Profit=[]
        self.Profit_rate=[]
        self.Capital_rate=[]  ## 以1為本金的歷次完整交易的本利和清單
        ## 未平倉
        self.OpenInterestQty=0
        self.OpenInterest=[]
        ## 交易紀錄總計
        self.TradeRecord=[]
        ## 成本
        self.G_spread=G_spread
        self.G_tax=G_tax
        self.G_commission=G_commission
        
        
    # 進場紀錄
    def Order(self, BS,Product,OrderTime,OrderPrice,OrderQty):
        if BS=='B' or BS=='Buy':
            for i in range(OrderQty):
                self.OpenInterest.append([1,Product,OrderTime,OrderPrice])
                self.OpenInterestQty +=1
        elif BS=='S' or BS=='Sell':
            for i in range(OrderQty):
                self.OpenInterest.append([-1,Product,OrderTime,OrderPrice])
                self.OpenInterestQty -=1
    # 出場紀錄(買賣別需與進場相反，多單進場則空單出場)
    def Cover(self, BS,Product,CoverTime,CoverPrice,CoverQty):
        if BS=='S' or BS=='Sell':
            for i in range(CoverQty):
                # 取得多單未平倉部位
                TmpInterest=[ i for i in self.OpenInterest if i[0]==1 ][0]
                if TmpInterest != []:
                    # 清除未平倉紀錄
                    self.OpenInterest.remove(TmpInterest)
                    self.OpenInterestQty -=1
                    # 新增交易紀錄
                    self.TradeRecord.append(['B',TmpInterest[1],TmpInterest[2],TmpInterest[3],CoverTime,CoverPrice])
                    if self.isFuture and Product=='TXF':
                        #self.Profit.append(CoverPrice*(1-self.G_commission-self.G_tax)-TmpInterest[3]*(1-self.G_commission-self.isFuture*self.G_tax))
                        self.Profit.append(CoverPrice*(1-self.G_commission/(200*CoverPrice)-self.G_tax)-TmpInterest[3]*(1-self.G_commission/(200*TmpInterest[3])-self.isFuture*self.G_tax))
                        #self.Profit.append((CoverPrice-TmpInterest[3])*(1-self.G_commission-self.G_tax))  ## 股票與期貨的不同.
                        self.Profit_rate.append((CoverPrice*(1-self.G_commission/(200*CoverPrice)-self.G_tax)-TmpInterest[3]*(1-self.G_commission/(200*TmpInterest[3])-self.isFuture*self.G_tax))/(TmpInterest[3]*(1-self.G_commission/(200*TmpInterest[3])-self.isFuture*self.G_tax))) ## 
                        #self.Profit_rate.append((CoverPrice*(1-self.G_commission-self.G_tax)-TmpInterest[3]*(1+self.G_commission+self.isFuture*self.G_tax))/TmpInterest[3]*(1+self.G_commission+self.isFuture*self.G_tax)) ## 
                        self.Capital_rate.append(1+(CoverPrice*(1-self.G_commission/(200*CoverPrice)-self.G_tax)-TmpInterest[3]*(1-self.G_commission/(200*TmpInterest[3])-self.isFuture*self.G_tax))/(TmpInterest[3]*(1-self.G_commission/(200*TmpInterest[3])-self.isFuture*self.G_tax)))  ## 初始財富為 "1倍" 之歷次完整交易的本利和清單.  CoverPrice與TmpInterest[3]的單位: 點數 or TWD. 
                        #self.Capital_rate.append(1+(CoverPrice*(1-self.G_commission-self.G_tax)-TmpInterest[3]*(1+self.G_commission+self.isFuture*self.G_tax))/TmpInterest[3]*(1+self.G_commission+self.isFuture*self.G_tax))  ## 初始財富為 "1倍" 之歷次完整交易的本利和清單.  CoverPrice與TmpInterest[3]的單位: 點數 or TWD.
                    elif self.isFuture and Product=='MXF':
                        self.Profit.append(CoverPrice*(1-self.G_commission/(50*CoverPrice)-self.G_tax)-TmpInterest[3]*(1-self.G_commission/(50*TmpInterest[3])-self.isFuture*self.G_tax))
                        self.Profit_rate.append((CoverPrice*(1-self.G_commission/(50*CoverPrice)-self.G_tax)-TmpInterest[3]*(1-self.G_commission/(50*TmpInterest[3])-self.isFuture*self.G_tax))/(TmpInterest[3]*(1-self.G_commission/(50*TmpInterest[3])-self.isFuture*self.G_tax))) ## 
                        self.Capital_rate.append(1+(CoverPrice*(1-self.G_commission/(50*CoverPrice)-self.G_tax)-TmpInterest[3]*(1-self.G_commission/(50*TmpInterest[3])-self.isFuture*self.G_tax))/(TmpInterest[3]*(1-self.G_commission/(50*TmpInterest[3])-self.isFuture*self.G_tax)))  ## 初始財富為 "1倍" 之歷次完整交易的本利和清單.  CoverPrice與TmpInterest[3]的單位: 點數 or TWD. 
                    elif self.isFuture and Product!='TXF' and Product!='MXF':
                        self.Profit.append(CoverPrice*(1-self.G_commission/(CoverPrice)-self.G_tax)-TmpInterest[3]*(1-self.G_commission/(TmpInterest[3])-self.isFuture*self.G_tax))
                        self.Profit_rate.append((CoverPrice*(1-self.G_commission/(CoverPrice)-self.G_tax)-TmpInterest[3]*(1-self.G_commission/(TmpInterest[3])-self.isFuture*self.G_tax))/(TmpInterest[3]*(1-self.G_commission/(TmpInterest[3])-self.isFuture*self.G_tax))) ## 
                        self.Capital_rate.append(1+(CoverPrice*(1-self.G_commission/(CoverPrice)-self.G_tax)-TmpInterest[3]*(1-self.G_commission/(TmpInterest[3])-self.isFuture*self.G_tax))/(TmpInterest[3]*(1-self.G_commission/(TmpInterest[3])-self.isFuture*self.G_tax)))  ## 初始財富為 "1倍" 之歷次完整交易的本利和清單.  CoverPrice與TmpInterest[3]的單位: 點數 or TWD.           
                    else:
                        #self.Profit.append(CoverPrice*(1-self.G_commission-self.G_tax)-TmpInterest[3]*(1-self.G_commission-self.isFuture*self.G_tax))
                        self.Profit.append(CoverPrice*(1-self.G_commission-self.G_tax)-TmpInterest[3]*(1-self.G_commission-self.isFuture*self.G_tax))
                        #self.Profit.append((CoverPrice-TmpInterest[3])*(1-self.G_commission-self.G_tax))  ## 股票與期貨的不同.
                        self.Profit_rate.append((CoverPrice*(1-self.G_commission-self.G_tax)-TmpInterest[3]*(1-self.G_commission-self.isFuture*self.G_tax))/(TmpInterest[3]*(1-self.G_commission-self.isFuture*self.G_tax))) ## 
                        #self.Profit_rate.append((CoverPrice*(1-self.G_commission-self.G_tax)-TmpInterest[3]*(1+self.G_commission+self.isFuture*self.G_tax))/TmpInterest[3]*(1+self.G_commission+self.isFuture*self.G_tax)) ## 
                        self.Capital_rate.append(1+(CoverPrice*(1-self.G_commission-self.G_tax)-TmpInterest[3]*(1-self.G_commission-self.isFuture*self.G_tax))/(TmpInterest[3]*(1-self.G_commission-self.isFuture*self.G_tax)))  ## 初始財富為 "1倍" 之歷次完整交易的本利和清單.  CoverPrice與TmpInterest[3]的單位: 點數 or TWD. 
                        #self.Capital_rate.append(1+(CoverPrice*(1-self.G_commission-self.G_tax)-TmpInterest[3]*(1+self.G_commission+self.isFuture*self.G_tax))/TmpInterest[3]*(1+self.G_commission+self.isFuture*self.G_tax))  ## 初始財富為 "1倍" 之歷次完整交易的本利和清單.  CoverPrice與TmpInterest[3]的單位: 點數 or TWD.
                else:
                    print('尚無進場')
        elif BS=='B' or BS=='Buy':
            for i in range(CoverQty):
                # 取得空單未平倉部位
                TmpInterest=[ i for i in self.OpenInterest if i[0]==-1 ][0]
                if TmpInterest != []:
                    # 清除未平倉紀錄
                    self.OpenInterest.remove(TmpInterest)
                    self.OpenInterestQty +=1
                    # 新增交易紀錄
                    self.TradeRecord.append(['S',TmpInterest[1],TmpInterest[2],TmpInterest[3],CoverTime,CoverPrice])
                    
                    if self.isFuture and Product=='TXF':
                        self.Profit.append(-CoverPrice*(1-self.G_commission/(200*CoverPrice)-self.G_tax)+TmpInterest[3]*(1-self.G_commission/(200*TmpInterest[3])-self.isFuture*self.G_tax))
                        self.Profit_rate.append((-CoverPrice*(1-self.G_commission/(200*CoverPrice)-self.G_tax)+TmpInterest[3]*(1-self.G_commission/(200*TmpInterest[3])-self.isFuture*self.G_tax))/(TmpInterest[3]*(1-self.G_commission/(200*TmpInterest[3])-self.isFuture*self.G_tax))) ## 
                        self.Capital_rate.append(1+(-CoverPrice*(1-self.G_commission/(200*CoverPrice)-self.G_tax)+TmpInterest[3]*(1-self.G_commission/(200*TmpInterest[3])-self.isFuture*self.G_tax))/(TmpInterest[3]*(1-self.G_commission/(200*TmpInterest[3])-self.isFuture*self.G_tax)))  ## 初始財富為 "1倍" 之歷次完整交易的本利和清單.  CoverPrice與TmpInterest[3]的單位: 點數 or TWD. 
                    elif self.isFuture and Product=='MXF':
                        self.Profit.append(-CoverPrice*(1-self.G_commission/(50*CoverPrice)-self.G_tax)+TmpInterest[3]*(1-self.G_commission/(50*TmpInterest[3])-self.isFuture*self.G_tax))
                        self.Profit_rate.append((-CoverPrice*(1-self.G_commission/(50*CoverPrice)-self.G_tax)+TmpInterest[3]*(1-self.G_commission/(50*TmpInterest[3])-self.isFuture*self.G_tax))/(TmpInterest[3]*(1-self.G_commission/(50*TmpInterest[3])-self.isFuture*self.G_tax))) ## 
                        self.Capital_rate.append(1+(-CoverPrice*(1-self.G_commission/(50*CoverPrice)-self.G_tax)+TmpInterest[3]*(1-self.G_commission/(50*TmpInterest[3])-self.isFuture*self.G_tax))/(TmpInterest[3]*(1-self.G_commission/(50*TmpInterest[3])-self.isFuture*self.G_tax)))  ## 初始財富為 "1倍" 之歷次完整交易的本利和清單.  CoverPrice與TmpInterest[3]的單位: 點數 or TWD. 
                    elif self.isFuture and Product!='TXF' and Product!='MXF':
                        self.Profit.append(-CoverPrice*(1-self.G_commission/(CoverPrice)-self.G_tax)+TmpInterest[3]*(1-self.G_commission/(TmpInterest[3])-self.isFuture*self.G_tax))
                        self.Profit_rate.append((-CoverPrice*(1-self.G_commission/(CoverPrice)-self.G_tax)+TmpInterest[3]*(1-self.G_commission/(TmpInterest[3])-self.isFuture*self.G_tax))/(TmpInterest[3]*(1-self.G_commission/(TmpInterest[3])-self.isFuture*self.G_tax))) ## 
                        self.Capital_rate.append(1+(-CoverPrice*(1-self.G_commission/(CoverPrice)-self.G_tax)+TmpInterest[3]*(1-self.G_commission/(TmpInterest[3])-self.isFuture*self.G_tax))/(TmpInterest[3]*(1-self.G_commission/(TmpInterest[3])-self.isFuture*self.G_tax)))  ## 初始財富為 "1倍" 之歷次完整交易的本利和清單.  CoverPrice與TmpInterest[3]的單位: 點數 or TWD.           
                    else:
                        self.Profit.append(-CoverPrice*(1-self.G_commission-self.G_tax)+TmpInterest[3]*(1-self.G_commission-self.isFuture*self.G_tax))
                        self.Profit_rate.append((-CoverPrice*(1-self.G_commission-self.G_tax)+TmpInterest[3]*(1-self.G_commission-self.isFuture*self.G_tax))/(TmpInterest[3]*(1-self.G_commission-self.isFuture*self.G_tax))) ## 
                        self.Capital_rate.append(1+(-CoverPrice*(1-self.G_commission-self.G_tax)+TmpInterest[3]*(1-self.G_commission-self.isFuture*self.G_tax))/(TmpInterest[3]*(1-self.G_commission-self.isFuture*self.G_tax)))  ## 初始財富為 "1倍" 之歷次完整交易的本利和清單.  CoverPrice與TmpInterest[3]的單位: 點數 or TWD. 
                        
                    
                else:
                    print('尚無進場')



# 取得當前未平倉數量
def GetOpenInterest(OpenInterestQty):               
    # 取得未平倉量
    return OpenInterestQty
# 取得交易紀錄清單(要進場與出場之後, 才會列入至此)
def GetTradeRecord(TradeRecord):               
    # 取得未平倉量
    return TradeRecord   
# 取得交易盈虧清單
def GetProfit(Profit):       
    return Profit 
# 取得交易投資報酬率清單
def GetProfitRate(Profit_rate):       
    return Profit_rate

# # 將股票的回測紀錄寫入MicroTest當中
# def StockMicroTestRecord(self,StrategyName,Discount):
#     microtest_db.login('jack','1234','ftserver.haohaninfo.com')
#     for row in self.TradeRecord:
#         Fee=row[3]*1000*0.001425*Discount + row[5]*1000*0.001425*Discount 
#         Tax=row[5]*1000*0.003
#         microtest_db.insert_to_server_db(   \
#         row[1],                             \
#         row[2].strftime('%Y-%m-%d'),        \
#         row[2].strftime('%H:%M:%S'),        \
#         row[3],                             \
#         row[0],                             \
#         '1',                                \
#         row[4].strftime('%Y-%m-%d'),        \
#         row[4].strftime('%H:%M:%S'),        \
#         row[5],                             \
#         Tax,                                \
#         Fee,                                \
#         StrategyName)    
#     microtest_db.commit()
# 將期貨的回測紀錄寫入MicroTest當中
# def FutureMicroTestRecord(self,StrategyName,ProductValue,Fee):
#     microtest_db.login('jack','1234','ftserver.haohaninfo.com')
#     for row in self.TradeRecord:
#         Tax=row[5]*ProductValue*0.00002*2
#         microtest_db.insert_to_server_db(   \
#         row[1],                             \
#         row[2].strftime('%Y-%m-%d'),        \
#         row[2].strftime('%H:%M:%S'),        \
#         str(int(row[3])),                             \
#         str(row[0]),                             \
#         '1',                                \
#         row[4].strftime('%Y-%m-%d'),        \
#         row[4].strftime('%H:%M:%S'),        \
#         str(row[5]),                             \
#         str(int(Tax)),                                \
#         str(int(Fee)),                                \
#         StrategyName) 
#     microtest_db.commit()
    
    
# def FutureMicroTestRecord(self,StrategyName,ProductValue,Fee,volume,account,password):
#     #microtest_db.login('jack','1234','ftserver.haohaninfo.com')
#     microtest_db.login(account,password,'140.128.36.207')
#     #microtest_db.login(account,password,'140.128.36.207')
#     for row in self.TradeRecord:
#         Tax=(row[3]+row[5])*ProductValue*0.00002
#         microtest_db.insert_to_server_db(row[1],row[2].strftime('%Y-%m-%d'),row[2].strftime('%H:%M:%S'),str(int(row[3])),str(row[0]),str(volume),row[4].strftime('%Y-%m-%d'),row[4].strftime('%H:%M:%S'),str(row[5]),str(int(Tax)),str(int(Fee)),StrategyName) 
#     microtest_db.commit()




# 取得交易總盈虧
def GetTotalProfit(Profit):  
    return sum(Profit)
# 取得交易次數
def GetTotalNumber(Profit):  
    return len(Profit)
# 取得平均交易盈虧(每次)
def GetAverageProfit(Profit): 
    return sum(Profit)/len(Profit)
# 取得交易 "平均" 投資報酬率
def GetAverageProfitRate(Profit_rate):       
    return sum(Profit_rate)/len(Profit_rate)
# 取得勝率=賺錢的交易次數/總交易次數
def GetWinRate(Profit):
    WinProfit = [ i for i in Profit if i > 0 ]
    return len(WinProfit)/len(Profit)
# 最大連續虧損(TWD,點數)
def GetAccLoss(Profit):
    AccLoss = 0
    MaxAccLoss = 0
    for p in Profit:
        if p <= 0:
            AccLoss+=p
            if AccLoss < MaxAccLoss:
                MaxAccLoss=AccLoss
        else:
            AccLoss=0
    return MaxAccLoss

# 最大 "歷次交易累計盈虧(TWD,點數)" 回落: 畫圖時: y軸是 "盈虧(Profit)", x軸是歷次交易的編號
def GetMDD_Profit(Profit):
    MDD, profit, MaxProfit = 0,0,0
    for p in Profit:
        profit += p
        MaxProfit = max(MaxProfit,profit)
        DD = MaxProfit - profit
        MDD = max(MDD,DD)
    return MDD

# 最大 "歷次交易累計投資報酬率" 回落: 畫圖時: y軸是 "投資報酬率(Profit_rate)", x軸是歷次交易的編號
def GetMDD_Profit_rate(Profit_rate):
    MDD,profit_rate,MaxProfit_rate = 0,0,0
    for p in Profit_rate:
        profit_rate += p
        MaxProfit_rate = max(MaxProfit_rate,profit_rate)
        DD = MaxProfit_rate - profit_rate
        MDD = max(MDD,DD)
    return MDD


# 最大的 "累計本利和(以1為初始本金)" 回落: 畫圖時: y軸是 "累計本利和(以1為初始本金)", x軸是歷次交易的編號
def GetMDD_CapitalRate1(CumulativeCapitalRate_series):
    MDD = 0
    #CumulativeCapitalRate_series, _ = self.GetCumulativeCapitalRate_finalReturn()
    maxval = CumulativeCapitalRate_series[0]
    for i in range(0,len(CumulativeCapitalRate_series),1):
        maxval=max(CumulativeCapitalRate_series[i],maxval)
        DD = maxval - CumulativeCapitalRate_series[i]
        MDD = max(MDD,DD)
    return MDD


# 最大的 "減少的累計本利和/最大累計本利和": 畫圖時: y軸是 "減少的累計本利和/最大累計本利和(浮動)", x軸是歷次交易的編號
def GetMDD_CapitalRate2(CumulativeCapitalRate_series):
    MDD = 0
    #CumulativeCapitalRate_series, _ = self.GetCumulativeCapitalRate_finalReturn()
    maxval = CumulativeCapitalRate_series[0]
    for i in range(0,len(CumulativeCapitalRate_series),1):
        maxval=max(CumulativeCapitalRate_series[i],maxval)
        temp=1.0-CumulativeCapitalRate_series[i]/maxval  ## = (maxval-CumulativeCapitalRate_series[i])/maxval=減少的累計本利和/最大累計本利和
        if(temp>MDD):
            MDD=temp
    return MDD




# 平均獲利(只看獲利的) 
def GetAverEarn(Profit):
    WinProfit = [ i for i in Profit if i > 0 ]
    return sum(WinProfit)/len(WinProfit)
# 平均虧損(只看虧損的)
def GetAverLoss(Profit):
    FailProfit = [ i for i in Profit if i < 0 ]
    return sum(FailProfit)/len(FailProfit)

# 歷次交易累計盈虧(單位: TWD,點數)清單: 買Order_Quantity "口(for期貨)" or "股(for股票,不是張)" 的歷次交易盈虧累計的清單. 
def GetCumulativeProfit(Profit):
    TotalProfit=[0]
    for i in Profit:
        TotalProfit.append(TotalProfit[-1]+i)
    return TotalProfit

# 歷次交易累計投資報酬率清單: 買Order_Quantity "口(for期貨)" or "股(for股票,不是張)" 的歷次交易投資報酬率累計的清單 
def GetCumulativeProfit_rate(Profit_rate):
    TotalProfit_rate=[0]
    for i in Profit_rate:
        TotalProfit_rate.append(TotalProfit_rate[-1]+i)
    return TotalProfit_rate

### 歷次交易累計本利和清單(以1為初始資金) & 最後總報酬率: 買Order_Quantity "口(for期貨)" or "股(for股票,不是張)". 
## 累計本利和清單(以1為初始資金): 至第i次交易(完成入出場)時,將所有1至i次所有完成入場與出場的交易的本利和相乘
## 最後總報酬率即為(以1為初始資金):所有完成入場與出場的交易的本利和相乘-1=累計本利和清單最後一個值-1. <說明>計算績效期間(不一定是第一個入場至最後一個出場期間),每根K棒期間的本利和複利計算=前一根K棒週期所得的本利和再度投入下一根K棒週期中計算本利和, 故為複利. 如果是沒有入場的期間, 此根K棒本利和=1+0, 如果是入場期間, 此根K棒本利和=1+每根K棒期間投資報酬率.
def GetCumulativeCapitalRate_finalReturn(Capital_rate):
    if len(Capital_rate) == 0:
        return [], 0  # 加這一行避免報錯

    CumulativeCapitalRate_series = []
    prod = 1
    for i in range(len(Capital_rate)):
        prod *= Capital_rate[i]
        CumulativeCapitalRate_series.append(prod)
    final_return = CumulativeCapitalRate_series[-1] - 1
    return CumulativeCapitalRate_series, final_return




# 產出交易績效圖(畫出 "累計盈虧(TWD,點數)清單")
def GeneratorProfitChart(Profit,StrategyName='Strategy'):
    
    # 計算累計績效
    TotalProfit = GetCumulativeProfit(Profit)
    # TotalProfit=[0]
    # for i in self.Profit:
    #     TotalProfit.append(TotalProfit[-1]+i)
    
    # 定義圖表
    ax1 = plt.subplot(111)
    # 繪製圖形
    ax1.plot(TotalProfit, '-', linewidth=1 )
    #定義標頭
    ax1.set_title('Cumulative Profit')
    plt.show()    # 顯示繪製圖表
    plt.savefig(StrategyName+'.png') #儲存繪製圖表
    
# 產出累計本利和(複利,(以1為初始資金))交易績效圖(畫出 "累計本利和清單(以1為初始資金)")
def GeneratorCapitalRateChart(CumulativeCapitalRate_series,StrategyName='Strategy'):
    ## 計算累計本利和rate(複利,(以1為初始資金))
    #CumulativeCapitalRate_series, _ = self.GetCumulativeCapitalRate_finalReturn()
    # clone = self.Capital_rate.copy()
    # prod = 1
    # for i in range(0,len(self.Capital_rate),1):
    #     prod = prod*self.Capital_rate[i]
    #     clone[i] = prod 
   # 定義圖表
    ax1 = plt.subplot(111)
    
    # 繪製圖形
    ax1.plot(CumulativeCapitalRate_series  , '-', linewidth=1 )
    #定義標頭
    ax1.set_title('Cumulative Capital Rate(initial capital 1)')
    plt.show()    # 顯示繪製圖表
    plt.savefig(StrategyName+'.png') #儲存繪製圖表    
    

    

# # 市價委託單(預設非當沖、倉別自動)
# def OrderMKT(Broker,Product,BS,Qty,DayTrade='0',OrderType='A'):
#     # 送出交易委託
#     # print([Broker, Product, BS, '',str(Qty), "IOC", "MKT" ,str(DayTrade),OrderType])
#     #OrderNo=GOC.Order(Broker, Product, BS, '0',str(Qty), "IOC", "MKT" ,str(DayTrade),OrderType)
#     #print(OrderNo)
#     # 判斷是否委託成功(這邊以元富為例)
#     if OrderNo != '委託失敗':
#         while True:
#             # 取得成交帳務
#             MatchInfo=GOC.MatchAccount(Broker,OrderNo)
#             # 判斷是否成交
#             if MatchInfo != []:
#                 # 成交則回傳
#                 return MatchInfo[0].split(',')
#     else:
#         return False
            
     
            
# # 範圍市價單(預設非當沖、倉別自動、掛上下N檔價1-5[預設3]、N秒尚未成交刪單[預設10])
# def OrderRangeMKT(Broker,Product,BS, Qty,DayTrade='0',OrderType='A',OrderPriceLevel=3,Wait=10): 
#     # 新增訂閱要下單的商品，預防沒有取到該商品報價
#     # GOC.AddQuote(Broker,Product,True)
#     # 取得委託商品的上下五檔來進行限價委託(這邊預設下單與報價使用同一個券商，若不同則需另外調整)
#     UpdnInfo=GOQ.SubscribeLast(Broker,'updn5',Product)
#     # 如果是買單，則掛上五檔委託
#     if BS == 'B':
#         OrderPoint=UpdnInfo[OrderPriceLevel*2]
#     elif BS == 'S':
#         OrderPoint=UpdnInfo[10+OrderPriceLevel*2]
#     # 送出交易委託
#     print([Broker, Product, BS, str(OrderPoint), str(Qty), "ROD", "LMT" ,str(DayTrade),OrderType])
#     OrderNo=GOC.Order(Broker, Product, BS, str(OrderPoint), str(Qty), "ROD", "LMT" ,str(DayTrade),OrderType )
#     # 設定刪單時間
#     EndTime=time.time()+Wait
#     # 判斷是否委託成功(這邊以元富為例)
#     if OrderNo != '委託失敗':
#         # 若大於刪單時間則跳出迴圈
#         while time.time() < EndTime:
#             # 取得成交帳務
#             MatchInfo=GOC.MatchAccount(Broker,OrderNo)
#             # 判斷是否成交
#             if MatchInfo != []:
#                 # 成交則回傳
#                 return MatchInfo[0].split(',')
#             # 稍等0.5秒
#             time.sleep(0.5)
#             print('尚未成交')
#         # 刪單並確認委託成功刪除
#         GOC.Delete(Broker,OrderNo)
#         GOC.GetAccount(Broker,OrderNo)
#         print('到期刪單')
#         return False
#     else:
#         return False 

# # 範圍市價單(預設非當沖、倉別自動、掛上下N檔價1-5[預設3]、N秒尚未成交刪單[預設10])
# def RangeMKTDeal(Broker,Product,BS, Qty,DayTrade='0',OrderType='A',OrderPriceLevel=3,Wait=10):
#     # 防止例外狀況，最多下三次單
#     for i in range(3):
#         OrderInfo=OrderRangeMKT(Broker,Product,BS,Qty,DayTrade,OrderType,OrderPriceLevel,Wait)
#         if OrderInfo != False:
#             return OrderInfo
#     # 三次委託皆失敗，建議當日不做交易
#     return False
