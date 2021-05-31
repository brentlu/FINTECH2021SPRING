import numpy as np

# A simple greedy approach
def myActionSimple(priceMat, transFeeRate):
    # Explanation of my approach:
	# 1. Technical indicator used: Watch next day price
	# 2. if next day price > today price + transFee ==> buy
    #       * buy the best stock
	#    if next day price < today price + transFee ==> sell
    #       * sell if you are holding stock
    # 3. You should sell before buy to get cash each day
    # default
    cash = 1000
    hold = 0
    # user definition
    nextDay = 1
    dataLen, stockCount = priceMat.shape  # day size & stock count   
    stockHolding = np.zeros((dataLen,stockCount))  # Mat of stock holdings
    actionMat = []  # An k-by-4 action matrix which holds k transaction records.
    
    for day in range( 0, dataLen-nextDay ) :
        dayPrices = priceMat[day]  # Today price of each stock
        nextDayPrices = priceMat[ day + nextDay ]  # Next day price of each stock
        
        if day > 0:
            stockHolding[day] = stockHolding[day-1]  # The stock holding from the previous action day
        
        buyStock = -1  # which stock should buy. No action when is -1
        buyPrice = 0  # use how much cash to buy
        sellStock = []  # which stock should sell. No action when is null
        sellPrice = []  # get how much cash from sell
        bestPriceDiff = 0  # difference in today price & next day price of "buy" stock
        stockCurrentPrice = 0  # The current price of "buy" stock
        
        # Check next day price to "sell"
        for stock in range(stockCount) :
            todayPrice = dayPrices[stock]  # Today price
            nextDayPrice = nextDayPrices[stock]  # Next day price
            holding = stockHolding[day][stock]  # how much stock you are holding
            
            if holding > 0 :  # "sell" only when you have stock holding
                if nextDayPrice < todayPrice*(1+transFeeRate) :  # next day price < today price, should "sell"
                    sellStock.append(stock)
                    # "Sell"
                    sellPrice.append(holding * todayPrice)
                    cash = holding * todayPrice*(1-transFeeRate) # Sell stock to have cash
                    stockHolding[day][sellStock] = 0
        
        # Check next day price to "buy"
        if cash > 0 :  # "buy" only when you have cash
            for stock in range(stockCount) :
                todayPrice = dayPrices[stock]  # Today price
                nextDayPrice = nextDayPrices[stock]  # Next day price
                
                if nextDayPrice > todayPrice*(1+transFeeRate) :  # next day price > today price, should "buy"
                    diff = nextDayPrice - todayPrice*(1+transFeeRate)
                    if diff > bestPriceDiff :  # this stock is better
                        bestPriceDiff = diff
                        buyStock = stock
                        stockCurrentPrice = todayPrice
            # "Buy" the best stock
            if buyStock >= 0 :
                buyPrice = cash
                stockHolding[day][buyStock] = cash*(1-transFeeRate) / stockCurrentPrice # Buy stock using cash
                cash = 0
                
        # Save your action this day
        if buyStock >= 0 or len(sellStock) > 0 :
            action = []
            if len(sellStock) > 0 :
                for i in range( len(sellStock) ) :
                    action = [day, sellStock[i], -1, sellPrice[i]]
                    actionMat.append( action )
            if buyStock >= 0 :
                action = [day, -1, buyStock, buyPrice]
                actionMat.append( action )
    return actionMat

# A DP-based approach to obtain the optimal return which allows
# consecutive 'cashDays' days to hold all cash without any stocks
# starting from day 'cashStart'
def myActionForceCash(priceMat, transFeeRate, cashStart, cashDays):
    # default
    initCash = 1000
    # user definition
    dayNum, stockNum = priceMat.shape  # day size & stock count
    dp = np.zeros((dayNum, stockNum+1), dtype=float)
    bt = np.zeros((dayNum, stockNum+1), dtype=int)
    actionMat = []  # An k-by-4 action matrix which holds k transaction records.

    # init the dp and bt matrix
    day = 0
    dayPrices = priceMat[day]  # Today price of each stock

    for stock in range(0, stockNum):
        dp[day][stock] = initCash*(1-transFeeRate)/dayPrices[stock]
        bt[day][stock] = -1

    dp[day][stockNum] = initCash
    bt[day][stockNum] = -1

    # compute the dp and bt matrix
    for day in range(1, dayNum):
        dayPrices = priceMat[day]  # Today price of each stock

        for cashTo in range(0, stockNum+1):
            if cashStart >= 0 and day >= cashStart and day < (cashStart+cashDays):
                # forced to select cash today
                buyIn = dp[day-1][stockNum]

                if cashTo != stockNum:
                    # buy new stock
                    buyIn = buyIn*(1-transFeeRate)/dayPrices[cashTo]

                if buyIn > dp[day][cashTo]:
                    dp[day][cashTo] = buyIn
                    bt[day][cashTo] = stockNum
            else:
                for cashFrom in range(0, stockNum+1):
                    cashOutValue = dp[day-1][cashFrom]

                    if cashFrom != stockNum and cashFrom != cashTo:
                        # sell stock
                        cashOutValue = cashOutValue*dayPrices[cashFrom]*(1-transFeeRate)

                    buyIn = cashOutValue

                    if cashTo != stockNum and cashFrom != cashTo:
                        # buy new stock
                        buyIn = buyIn*(1-transFeeRate)/dayPrices[cashTo]

                    if buyIn > dp[day][cashTo]:
                        dp[day][cashTo] = buyIn
                        bt[day][cashTo] = cashFrom

    #for day in range(0, dayNum):
    #    print(dp[day])
    #    print(bt[day])

    # backtrace
    holding = stockNum

    cashDaysTotal = 0
    cashDaysCount = 0
    cashDaysContinue = 0

    for day in range(dayNum-1, -1, -1):
        dayPrices = priceMat[day]  # Today price of each stock

        sellStock = -1
        buyStock = -1
        transValue = 0

        if holding == stockNum:
            cashDaysTotal += 1
            cashDaysCount += 1
        else:
            if cashDaysCount > cashDaysContinue:
                cashDaysContinue = cashDaysCount
            cashDaysCount = 0

        prevHolding = bt[day][holding]

        if holding != stockNum and holding != prevHolding:
            buyStock = holding
            transValue = dp[day][buyStock]*dayPrices[buyStock]/(1-transFeeRate)

        if prevHolding != stockNum and holding != prevHolding and prevHolding != -1:
            sellStock = prevHolding
            transValue = dp[day][sellStock]*dayPrices[sellStock]

        if sellStock != -1 or buyStock != -1:
            actionMat.append([day, sellStock, buyStock, transValue])

        holding = prevHolding
        if holding == -1:
            cashDaysTotal += day
            cashDaysCount += day
            if cashDaysCount > cashDaysContinue:
                cashDaysContinue = cashDaysCount
            break

    actionMat.reverse()

    #for action in actionMat:
    #    print(action)

    #print('Submit: rr=%f%%' % ((actionMat[-1][3]-initCash)*100/initCash))
    #print('Submit: Non continueous cash holding=%d' % (cashDaysTotal))
    #print('Submit: Continueous cash holding=%d' %(cashDaysContinue))

    return actionMat

# A DP-based approach to obtain the optimal return
def myAction01(priceMat, transFeeRate):
    # no all-cash window needed
    return myActionForceCash(priceMat, transFeeRate, -1, -1)

# An approach that allow non-consecutive K days to hold all cash without any stocks
def myAction02(priceMat, transFeeRate, K):
    # default
    initCash = 1000
    # user definition
    dayNum, stockNum = priceMat.shape  # day size & stock count
    dp = np.zeros((K+1, dayNum, stockNum+1), dtype=float)
    bt = np.zeros((K+1, dayNum, stockNum+1), dtype=int)
    actionMat = []  # An k-by-4 action matrix which holds k transaction records.

    # init the dp and bt matrix
    for cashDay in range(0, K+1):
        day = cashDay
        dayPrices = priceMat[day]  # Today price of each stock

        for stock in range(0, stockNum):
            dp[cashDay][day][stock] = initCash*(1-transFeeRate)/dayPrices[stock]
            bt[cashDay][day][stock] = -1

        dp[cashDay][day][stockNum] = initCash
        bt[cashDay][day][stockNum] = -1

    cashDay = 0
    for day in range(1+cashDay, dayNum):
        dp[cashDay][day][stockNum] = initCash
        bt[cashDay][day][stockNum] = -1

    # compute the dp and bt matrix
    for cashDay in range(0, K+1):
        for day in range(1+cashDay, dayNum):
            dayPrices = priceMat[day]  # Today price of each stock

            for cashTo in range(0, stockNum+1):
                if cashDay == 0 and cashTo == stockNum:
                    continue

                for cashFrom in range(0, stockNum+1):
                    if cashDay == 0 and cashFrom == stockNum:
                        continue

                    if cashTo == stockNum:
                        cashOutValue = dp[cashDay-1][day-1][cashFrom]
                    else:
                        cashOutValue = dp[cashDay][day-1][cashFrom]

                    if cashFrom != stockNum and cashFrom != cashTo:
                        # sell stock
                        cashOutValue = cashOutValue*dayPrices[cashFrom]*(1-transFeeRate)

                    buyIn = cashOutValue

                    if cashTo != stockNum and cashFrom != cashTo:
                        # buy new stock
                        buyIn = buyIn*(1-transFeeRate)/dayPrices[cashTo]

                    if buyIn > dp[cashDay][day][cashTo]:
                        dp[cashDay][day][cashTo] = buyIn
                        bt[cashDay][day][cashTo] = cashFrom

    # backtrace
    holding = stockNum
    cashDay = K

    cashDaysTotal = 0
    cashDaysCount = 0
    cashDaysContinue = 0

    for day in range(dayNum-1, -1, -1):
        dayPrices = priceMat[day]  # Today price of each stock

        sellStock = -1
        buyStock = -1
        transValue = 0

        if holding == stockNum:
            cashDaysTotal += 1
            cashDaysCount += 1
        else:
            if cashDaysCount > cashDaysContinue:
                cashDaysContinue = cashDaysCount
            cashDaysCount = 0

        prevHolding = bt[cashDay][day][holding]

        if holding != stockNum and holding != prevHolding:
            buyStock = holding
            transValue = dp[cashDay][day][buyStock]*dayPrices[buyStock]/(1-transFeeRate)

        if prevHolding != stockNum and holding != prevHolding and prevHolding != -1:
            sellStock = prevHolding
            if holding == stockNum:
                transValue = dp[cashDay-1][day][sellStock]*dayPrices[sellStock]
            else:
                transValue = dp[cashDay][day][sellStock]*dayPrices[sellStock]

        if sellStock != -1 or buyStock != -1:
            actionMat.append([day, sellStock, buyStock, transValue])

        if holding == stockNum:
            cashDay -= 1

        holding = prevHolding
        if holding == -1:
            cashDaysTotal += day
            cashDaysCount += day
            if cashDaysCount > cashDaysContinue:
                cashDaysContinue = cashDaysCount
            break

    actionMat.reverse()

    #for action in actionMat:
    #    print(action)

    #print("cashDay %d" % (cashDay))

    #print('Submit: rr=%f%%' % ((actionMat[-1][3]-initCash)*100/initCash))
    #print('Submit: Non continueous cash holding=%d' % (cashDaysTotal))
    #print('Submit: Continueous cash holding=%d' %(cashDaysContinue))

    return actionMat

# Use brute-force to find the K-day window to hold all cash without any stocks
def myAction03BruteForce(priceMat, transFeeRate, K):
    # user definition
    dayNum, stockNum = priceMat.shape  # day size & stock count

    bestActionMat = myActionForceCash(priceMat, transFeeRate, 0, K)
    bestStartDay = 0

    for day in range(1, dayNum):
        if day+K >= dayNum:
            break;

        actionMat = myActionForceCash(priceMat, transFeeRate, day, K)

        if actionMat[-1][3] > bestActionMat[-1][3]:
            bestActionMat = actionMat
            bestStartDay = day

    print("bestStartDay %d" % (bestStartDay))

    return bestActionMat

# An approach that allow consecutive K days to hold all cash without any stocks    
def myAction03(priceMat, transFeeRate, K):
    return myAction03BruteForce(priceMat, transFeeRate, K)
