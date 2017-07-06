import html

import numpy as np
import visdom


class Visdom(visdom.Visdom):
    """A Wrapper for Visdom
    
    This wrapper reduces the communication overhead with the visdom server by 
    pooling requests."""

    def __init__(self, *args, buffer_size=10, env=None, name=None, **kwargs):
        super(Visdom, self).__init__(*args, **kwargs)
        self.buffer_size = buffer_size
        self.env = env
        self.name = name
        self.wincache = {}
        self.buffer = {}

    def _process(self, f, *args, **kwargs):
        has_title = "opts" in kwargs and "title" in kwargs["opts"]

        if has_title:
            title = kwargs["opts"]["title"]

            if title in self.wincache:
                win = self.wincache[title]
                kwargs["win"] = win
        else:
            title = ""

        if has_title and self.name is not None:
            kwargs["opts"]["title"] = "[{}] ".format(self.name) + title

        if "env" not in kwargs:
            kwargs["env"] = self.env

        ret = f(*args, **kwargs)

        if has_title and title not in self.wincache:
            self.wincache[title] = ret

        return ret

    def text(self, text, *args, **kwargs):
        return self._process(super(Visdom, self).text, text, *args, **kwargs)

    def code(self, text, *args, **kwargs):
        text = html.escape(text)
        text = "<pre>{}</pre>".format(text)

        return self._process(super(Visdom, self).text, text, *args, **kwargs)

    def scatter(self, *args, **kwargs):
        return self._process(super(Visdom, self).scatter, *args, **kwargs)

    def plot(self, X, Y, opts=dict()):
        title = opts.get("title")

        if title not in self.buffer:
            self.buffer[title] = {"X": [], "Y": []}

        env = self.env

        self.buffer[title]["X"].extend(X)
        self.buffer[title]["Y"].extend(Y)

        if len(self.buffer[title]["X"]) > self.buffer_size:
            X = np.array(self.buffer[title]["X"])
            Y = np.array(self.buffer[title]["Y"])

            if title not in self.wincache:
                opts["title"] = title

                self.wincache[title] = self.line(
                    X=X,
                    Y=Y,
                    env=env,
                    opts=opts
                )
            else:
                self.line(
                    X=X,
                    Y=Y,
                    env=env,
                    win=self.wincache[title],
                    update="append"
                )

            del self.buffer[title]
