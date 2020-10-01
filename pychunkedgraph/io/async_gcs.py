from asyncio import new_event_loop

LOOP = new_event_loop()


async def get_files_task(bucket, files):
    from asyncio import gather
    from asyncio import create_task
    from aiohttp import ClientSession
    from gcloud.aio.storage import Storage

    tasks = []
    async with ClientSession() as session:
        client = Storage(session=session)
        for f in files:
            tasks.append(create_task(client.download(bucket, f)))
        return await gather(*tasks)


def get_files(bucket, files):
    return LOOP.run_until_complete(get_files_task(bucket, files))