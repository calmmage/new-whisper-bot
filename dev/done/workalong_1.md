- [x] add 'memory' parameter to pyrogram downloader
    - [x] download_file_with_pyrogram

```python

async def download(
        self,
        file_name: str = "",
        in_memory: bool = False,
        block: bool = True,
        progress: Callable = None,
        progress_args: tuple = ()
) -> str:
    """Bound method *download* of :obj:`~pyrogram.types.Message`.

    Use as a shortcut for:

    .. code-block:: python

        await client.download_media(message)

    Example:
        .. code-block:: python

            await message.download()

    Parameters:
        file_name (``str``, *optional*):
            A custom *file_name* to be used instead of the one provided by Telegram.
            By default, all files are downloaded in the *downloads* folder in your working directory.
            You can also specify a path for downloading files in a custom location: paths that end with "/"
            are considered directories. All non-existent folders will be created automatically.

        in_memory (``bool``, *optional*):
            Pass True to download the media in-memory.
            A binary file-like object with its attribute ".name" set will be returned.
            Defaults to False.

        block (``bool``, *optional*):
            Blocks the code execution until the file has been downloaded.
            Defaults to True.

        progress (``Callable``, *optional*):
            Pass a callback function to view the file transmission progress.
            The function must take *(current, total)* as positional arguments (look at Other Parameters below for a
            detailed description) and will be called back each time a new file chunk has been successfully
            transmitted.

        progress_args (``tuple``, *optional*):
            Extra custom arguments for the progress callback function.
            You can pass anything you need to be available in the progress callback scope; for example, a Message
            object or a Client instance in order to edit the message with the updated progress status.

    Other Parameters:
        current (``int``):
            The amount of bytes transmitted so far.

        total (``int``):
            The total size of the file.

        *args (``tuple``, *optional*):
            Extra custom arguments as defined in the ``progress_args`` parameter.
            You can either keep ``*args`` or add every single extra argument in your function signature.

    Returns:
        On success, the absolute path of the downloaded file as string is returned, None otherwise.

    Raises:
        RPCError: In case of a Telegram RPC error.
        ``ValueError``: If the message doesn't contain any downloadable media
    """
    return await self._client.download_media(
        message=self,
        file_name=file_name,
        in_memory=in_memory,
        block=block,
        progress=progress,
        progress_args=progress_args,
    )
```