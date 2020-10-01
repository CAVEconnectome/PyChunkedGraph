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
    from asyncio import new_event_loop
    from asyncio import get_running_loop

    try:
        # try using an exising event loop
        loop = get_running_loop()
        return loop.run_until_complete(get_files_task(bucket, files))
    except RuntimeError:
        # no event loop running, create and close
        loop = new_event_loop()
        resp = loop.run_until_complete(get_files_task(bucket, files))
        loop.close()
        return resp