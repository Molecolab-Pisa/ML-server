# The MIT License (MIT)
# 
# Copyright (c) 2018 Bernat Font Garcia
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE

def recvall(sock, dest):
    """Gets the data in dest. Dest is the empty data array

    Args:
       dest: Object to be read into.
    Raises:
       Disconnected: Raised if client is disconnected.
    Returns:
       The data read from the socket to be read into dest.
    """
    import numpy as np

    buf = np.zeros(0, np.byte)
    blen = dest.itemsize * dest.size
    if (blen > len(buf)):
        buf.resize(blen)
    bpos = 0

    while bpos < blen:
        timeout = False
        # post-2.5 version: slightly more compact for modern python versions
        try:
          bpart = 1
          bpart = sock.recv_into(buf[bpos:], blen-bpos)
        except socket.timeout:
          print(" @SOCKET:   Timeout in status recvall, trying again!")
          timeout = True
          pass
        if (not timeout and bpart == 0):
          raise Disconnected()
        bpos += bpart

    if np.isscalar(dest):
        return np.fromstring(buf[0:blen], dest.dtype)[0]
    else:
        return np.fromstring(buf[0:blen], dest.dtype).reshape(dest.shape)
