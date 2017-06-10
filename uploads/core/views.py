from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from uploads.core.models import Document
from uploads.core.forms import DocumentForm
import uploads.core.matching.kie_find_by_hash as finder
import os.path

from django.http import JsonResponse


def index(request):
    print(request.method)
    return render(request, 'core/index.html')


def upload(request):
    if request.method == 'POST' and request.FILES['file']:
        myfile = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        finds = finder.checkFromRAM('%s/%s' % (settings.MEDIA_ROOT, filename))
        notFound = False
        if len(finds) == 0:
            notFound = True
        return JsonResponse({
            'finds': finds,
            'not_found': notFound
        })

    return JsonResponse({})


def home(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        # finds = finder.check('%s/%s' % (settings.MEDIA_ROOT, filename), '/home/user/py-compare-images/images/hash/', True, 100,
        #               '.des.jpg')
        finds = finder.checkFromRAM('%s/%s' % (settings.MEDIA_ROOT, filename))
        print(finds)
        notFound = False
        if len(finds) == 0:
            notFound = True
        return render(request, 'core/simple_upload.html', {
            'finds': finds,
            'not_found': notFound
        })
    return render(request, 'core/simple_upload.html')
    # documents = Document.objects.all()
    # return render(request, 'core/home.html', { 'documents': documents })


def simple_upload(request):
    # if request.method == 'POST' and request.FILES['myfile']:
    #     myfile = request.FILES['myfile']
    #     fs = FileSystemStorage()
    #     filename = fs.save(myfile.name, myfile)
    #     uploaded_file_url = fs.url(filename)
    #     finds = check('%s/%s' % (settings.MEDIA_ROOT, filename), '/home/user/py-compare-images/images/hash/', True, 100,
    #                   '.des.jpg')
    #     print(finds)
    #     return render(request, 'core/simple_upload.html', {
    #         'finds': finds
    #     })
    # return render(request, 'core/simple_upload.html')
    return render(request, 'core/simple_upload.html')


def model_form_upload(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('home')
    else:
        form = DocumentForm()
    return render(request, 'core/model_form_upload.html', {
        'form': form
    })
