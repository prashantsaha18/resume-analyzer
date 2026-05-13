class FirebaseAuthMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
    def __call__(self, request):
        if request.method == "OPTIONS":
            from django.http import HttpResponse
            r = HttpResponse()
            r["Access-Control-Allow-Origin"]  = "*"
            r["Access-Control-Allow-Methods"] = "GET,POST,PUT,PATCH,DELETE,OPTIONS"
            r["Access-Control-Allow-Headers"] = "Authorization,Content-Type"
            return r
        return self.get_response(request)