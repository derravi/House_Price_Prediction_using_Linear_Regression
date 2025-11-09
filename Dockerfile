#Choose base image
FROM python:3.13.7

#Make one new Directory on docker root folder
RUN mkdir -p house_price_prediction

#Make a New Directory into inside the Docker
WORKDIR /house_price_prediction 

#Copy All the Currunt Directory file into the Docker Container.
COPY . /house_price_prediction

# Its create while Image is Building.
RUN pip install -r requirements.txt

#Set Enviornment Veriable
ENV PORT=8000
    
#Expose into the FastApi Localhost ip (Expose FastAPI’s port)
EXPOSE 8000

#Final Commant for running the Project.
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]