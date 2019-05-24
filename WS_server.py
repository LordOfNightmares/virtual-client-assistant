import asyncio
import json

import websockets

# from sqldb import *
from entity.conversation import Conversation
from entity.conversationrepo import ConversationDbRepo
from entity.message import Message
from entity.messagerepo import MessageDbRepo
from entity.user import User
from entity.userrepo import UserDbRepo

online_users = set()

async def prepare_conversation(data):
    cdb = ConversationDbRepo()
    c = cdb.get(data['cid'])
    if not c:
        print('--->',data)
        if data['uid']:
            c = Conversation(data['cid'], data['uid'])
            cdb.save(c)
    return c


async def process_msg(websocket):
    async for message in websocket:
        data = json.loads(message)
        if data:
            action = data['action']
            if action == "conversation_data":
                mdb = MessageDbRepo()
                m = mdb.last(data['cid'])
                print(m)
                c = await prepare_conversation(data)
                if c:
                    udb = UserDbRepo()
                    print("c",c)
                    u = udb.get(c.user_id)
                    await websocket.send(json.dumps({"answer": "user_confirmed", "user": dict(u)}))
                    messages = [dict(m) for m in mdb.all(data['cid'])]
                    await websocket.send(json.dumps({"answer": "messages_loaded", "messages": messages}))
            if action == "user_data":
                udb = UserDbRepo()
                u = User(data['user_first_name'], data['user_last_name'], data['user_email'], data['user_phone'])
                u_res = udb.find(u)
                print(u_res)
                if not u_res:
                    udb.save(u)
                else:
                    u = u_res
                await websocket.send(json.dumps({"answer": "user_confirmed", "user": dict(u)}))
                data['uid'] = u.get_id()
                await prepare_conversation(data)
            if action == "user_message":
                mdb = MessageDbRepo()
                # print(data['m'])
                m = Message('', data['user_message'], data['uid'], 0, data['cid'])
                mdb.save(m)
                await websocket.send(json.dumps({"answer": "message_confirmed", "message": dict(m)}))
                # udb.save(m)


# now = datetime.utcnow().isoformat() + 'Z'
# await websocket.send(now)
# await asyncio.sleep(random.random() * 3)


# def load_msg(cid):
#     mdb = MessageDbRepo()
#     msg = mdb.all(cid)
#     print(msg)


# def send_msg(message):
#     await asyncio.wait([ws.send("Hello!") for ws in online_users])

async def register(websocket):
    online_users.add(websocket)


async def unregister(websocket):
    online_users.remove(websocket)


async def test():
    await asyncio.wait([ws.send("test") for ws in online_users])


async def process(websocket, path):
    print("process")
    print(len(online_users))
    await register(websocket)
    await process_msg(websocket)
    try:
        await test()
        await asyncio.sleep(5)
    finally:
        await unregister(websocket)


if __name__ == '__main__':
    # sql = Sql('server.db')
    # logging.basicConfig(level=logging.DEBUG)
    start_server = websockets.serve(process, 'localhost', 9000)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
