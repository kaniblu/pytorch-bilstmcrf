class ImmutablePropertiesObject(object):
    def __init__(self, reserved=(), **feats):
        for f in reserved:
            if f in feats:
                raise ValueError("{} is a reserved attribute, "
                                 "therefore cannot be set as a "
                                 "feature.".format(f))

        self._feats = feats

    def __getitem__(self, item):
        return self._feats[item]

    def __getattribute__(self, item):
        getattr = super(ImmutablePropertiesObject, self).__getattribute__
        try:
            _feats = getattr("_feats")
        except:
            _feats = {}

        if item in _feats:
            return _feats[item]
        else:
            return getattr(item)

    @staticmethod
    def _privatize(attr):
        return "_{}".format(attr)

    @property
    def feats(self):
        return self._feats