import requests
import json

lst = [];

def replays(userID, ran):
        url = "https://halite.io/api/web/game"
        prev = ""
        rurl = "https://s3.amazonaws.com/halitereplaybucket/"
        for i in range(5):
                try:
                        open(userID + str(i*100000) + str(43434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343) + '.txt', 'r')
                        print i, "lol"
                        continue
                except IOError:
                        print i, "next"
                if (i == 0):
                        querystring = {"userID":userID,"limit":"10000"}
                else:
                        querystring = {"userID":userID,"limit":"10","startingID":prev}

                headers = {
                    'userid': "1017",
                    'limit': "10",
                    'startingid': "st",
                    'cache-control': "no-cache",
                    'postman-token': "d519fb4b-2b8a-3773-bd10-b8d3a48c2494"
            }

                response = requests.request("GET", url, headers=headers, params=querystring)
                global lst
                res = json.loads(response.text)
                print len(res)
                for j, r in enumerate(res):
			try:
                         #print r["replayName"]
                         lst.append(r)
                         prev = r["gameID"]
			 c = rurl + r["replayName"]
			 print c
                         replay = requests.request("GET", c, headers=headers)
                         f = open(r["replayName"] + '.hlt', 'w')
                         f.write(replay.text)
			except requests.exceptions.ConnectionError:
			 pass

print replays("2609", 6400)
#print replays("2557", 2290)
#print replays("1017", 1101)
#print replays("3157",5325)




