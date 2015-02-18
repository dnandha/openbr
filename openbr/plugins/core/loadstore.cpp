/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2012 The MITRE Corporation                                      *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License");           *
 * you may not use this file except in compliance with the License.          *
 * You may obtain a copy of the License at                                   *
 *                                                                           *
 *     http://www.apache.org/licenses/LICENSE-2.0                            *
 *                                                                           *
 * Unless required by applicable law or agreed to in writing, software       *
 * distributed under the License is distributed on an "AS IS" BASIS,         *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 * See the License for the specific language governing permissions and       *
 * limitations under the License.                                            *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/qtutils.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Caches transform training.
 * \author Josh Klontz \cite jklontz
 */
class LoadStoreTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString transformString READ get_transformString WRITE set_transformString RESET reset_transformString STORED false)
    Q_PROPERTY(QString fileName READ get_fileName WRITE set_fileName RESET reset_fileName STORED false)
    BR_PROPERTY(QString, transformString, "Identity")
    BR_PROPERTY(QString, fileName, QString())

public:
    Transform *transform;

    LoadStoreTransform() : transform(NULL) {}

    QString description(bool expanded = false) const
    {
        if (expanded) {
            QString res = transform->description(expanded);
            return res;
        }
        return br::Object::description(expanded);
    }

    Transform *simplify(bool &newTForm)
    {
        Transform *res = transform->simplify(newTForm);
        return res;
    }

    QList<Object *> getChildren() const
    {
        QList<Object *> rval;
        rval.append(transform);
        return rval;
    }
private:

    void init()
    {
        if (transform != NULL) return;
        if (fileName.isEmpty()) fileName = QRegExp("^[_a-zA-Z0-9]+$").exactMatch(transformString) ? transformString : QtUtils::shortTextHash(transformString);

        if (!tryLoad())
            transform = make(transformString);
        else
            trainable = false;
    }

    bool timeVarying() const
    {
        return transform->timeVarying();
    }

    void train(const QList<TemplateList> &data)
    {
        if (QFileInfo(getFileName()).exists())
            return;

        transform->train(data);

        qDebug("Storing %s", qPrintable(fileName));
        QtUtils::BlockCompression compressedOut;
        QFile fout(fileName);
        QtUtils::touchDir(fout);
        compressedOut.setBasis(&fout);

        QDataStream stream(&compressedOut);
        QString desc = transform->description();

        if (!compressedOut.open(QFile::WriteOnly))
            qFatal("Failed to open %s for writing.", qPrintable(file));

        stream << desc;
        transform->store(stream);
        compressedOut.close();
    }

    void project(const Template &src, Template &dst) const
    {
        transform->project(src, dst);
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        transform->project(src, dst);
    }

    void projectUpdate(const Template &src, Template &dst)
    {
        transform->projectUpdate(src, dst);
    }

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        transform->projectUpdate(src, dst);
    }

    void finalize(TemplateList &output)
    {
        transform->finalize(output);
    }

    QString getFileName() const
    {
        if (QFileInfo(fileName).exists()) return fileName;
        const QString file = Globals->sdkPath + "/share/openbr/models/transforms/" + fileName;
        return QFileInfo(file).exists() ? file : QString();
    }

    bool tryLoad()
    {
        const QString file = getFileName();
        if (file.isEmpty()) return false;

        qDebug("Loading %s", qPrintable(file));
        QFile fin(file);
        QtUtils::BlockCompression reader(&fin);
        if (!reader.open(QIODevice::ReadOnly)) {
            if (QFileInfo(file).exists()) qFatal("Unable to open %s for reading. Check file permissions.", qPrintable(file));
            else            qFatal("Unable to open %s for reading. File does not exist.", qPrintable(file));
        }

        QDataStream stream(&reader);
        stream >> transformString;

        transform = Transform::make(transformString);
        transform->load(stream);

        return true;
    }
};

BR_REGISTER(Transform, LoadStoreTransform)

} // namespace br

#include "core/loadstore.moc"
